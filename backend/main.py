from fastapi import FastAPI, Depends, HTTPException, WebSocket, WebSocketDisconnect
from sqlmodel import Session, select
from typing import List, Dict
import uuid
import json

from .database import engine, create_db_and_tables, get_session
from .models import Package, PackageBase, DeliveryRoute
from datetime import datetime, timedelta

app = FastAPI(title="Intelligent Package Tracking API", version="0.1.0")

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.on_event("startup")
def on_startup():
    create_db_and_tables()
    # Seed data if empty
    with Session(engine) as session:
        statement = select(Package)
        results = session.exec(statement)
        if not results.first():
            pkg = Package(
                order_id=f"PKG-{uuid.uuid4().hex[:8].upper()}",
                status="in_transit",
                weight=2.5,
                barcode="1234567890",
                eta=datetime.utcnow() + timedelta(hours=2)
            )
            session.add(pkg)
            session.commit()
            session.refresh(pkg)
            
            # Initial route point
            route = DeliveryRoute(
                package_id=pkg.id,
                latitude=-23.5505,
                longitude=-46.6333,
                speed=45.0
            )
            session.add(route)
            session.commit()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Package Tracking API"}

@app.post("/api/v1/packages", response_model=Package)
def create_package(package: PackageBase, session: Session = Depends(get_session)):
    db_package = Package.model_validate(package)
    session.add(db_package)
    session.commit()
    session.refresh(db_package)
    return db_package

@app.get("/api/v1/packages", response_model=List[Package])
def read_packages(session: Session = Depends(get_session)):
    packages = session.exec(select(Package)).all()
    return packages

@app.get("/api/v1/packages/{package_id}", response_model=Package)
def read_package(package_id: uuid.UUID, session: Session = Depends(get_session)):
    package = session.get(Package, package_id)
    if not package:
        raise HTTPException(status_code=404, detail="Package not found")
    return package

@app.get("/api/v1/tracking/{package_id}/history", response_model=List[DeliveryRoute])
def get_tracking_history(package_id: uuid.UUID, session: Session = Depends(get_session)):
    routes = session.exec(select(DeliveryRoute).where(DeliveryRoute.package_id == package_id)).all()
    return routes

# WebSocket for Real-time Updates
@app.websocket("/ws/notifications")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # We can receive filter commands from the frontend if needed
            data = await websocket.receive_text()
            # For now, just keep the connection alive
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.post("/api/v1/internal/broadcast")
async def internal_broadcast(payload: Dict):
    """
    Internal endpoint for the Edge Device (YOLO script) to push live updates.
    """
    await manager.broadcast(json.dumps(payload))
    return {"status": "broadcasted"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
