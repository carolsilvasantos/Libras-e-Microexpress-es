from sqlmodel import SQLModel, Field, Relationship
from typing import Optional, List, Dict
from datetime import datetime
import uuid

class PackageBase(SQLModel):
    order_id: str = Field(unique=True, index=True)
    status: str = Field(default="pending", index=True) # pending, in_transit, delivered, failed
    weight: Optional[float] = None
    dimensions_json: Optional[str] = None # JSON string
    barcode: Optional[str] = None
    recipient_id: Optional[uuid.UUID] = None
    eta: Optional[datetime] = None

class Package(PackageBase, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Relationships
    routes: List["DeliveryRoute"] = Relationship(back_populates="package")
    visual_analyses: List["VisualAnalysis"] = Relationship(back_populates="package")

class DeliveryRoute(SQLModel, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    package_id: uuid.UUID = Field(foreign_key="package.id")
    deliverer_id: Optional[uuid.UUID] = None
    latitude: float
    longitude: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    speed: Optional[float] = None
    
    package: Package = Relationship(back_populates="routes")

class VisualAnalysis(SQLModel, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    package_id: uuid.UUID = Field(foreign_key="package.id")
    image_s3_path: Optional[str] = None
    detected_objects_json: Optional[str] = None # JSON string
    integrity_status: str = Field(default="ok") # ok, damaged, open
    confidence_score: float
    processed_at: datetime = Field(default_factory=datetime.utcnow)
    
    package: Package = Relationship(back_populates="visual_analyses")
