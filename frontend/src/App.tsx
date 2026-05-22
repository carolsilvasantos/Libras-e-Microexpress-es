import React, { useState, useEffect } from 'react';
import { Package, MapPin, Search, Bell, Menu, User, Truck, CheckCircle, AlertCircle, Clock } from 'lucide-react';
import axios from 'axios';

interface PackageData {
  id: string;
  order_id: string;
  status: 'pending' | 'in_transit' | 'delivered' | 'failed';
  weight: number;
  eta: string;
}

function App() {
  const [packages, setPackages] = useState<PackageData[]>([]);
  const [selectedPackage, setSelectedPackage] = useState<PackageData | null>(null);
  const [notifications, setNotifications] = useState<any[]>([]);

  useEffect(() => {
    // 1. Initial Load
    const fetchPackages = async () => {
      try {
        // Mocking for now, will connect to http://localhost:8000/api/v1/packages later
        setPackages([
          { id: '1', order_id: 'PKG-A8F2E1', status: 'in_transit', weight: 2.5, eta: '2026-04-26T14:00:00Z' },
          { id: '2', order_id: 'PKG-B9C3D4', status: 'pending', weight: 1.2, eta: '2026-04-27T10:00:00Z' },
          { id: '3', order_id: 'PKG-C7D8E9', status: 'delivered', weight: 0.8, eta: '2026-04-25T16:30:00Z' },
        ]);
      } catch (err) {
        console.error("Error fetching packages", err);
      }
    };
    fetchPackages();

    // 2. WebSocket Connection
    const socket = new WebSocket('ws://localhost:8000/ws/notifications');
    
    socket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      console.log("Live Alert:", data);
      
      if (data.type === 'DETECTION') {
        setNotifications(prev => [data, ...prev].slice(0, 5));
      }
      
      if (data.type === 'TRACKING_UPDATE') {
        // Update package location in state if needed
      }
    };

    return () => socket.close();
  }, []);

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'in_transit': return <Truck className="text-primary w-5 h-5" />;
      case 'delivered': return <CheckCircle className="text-green-500 w-5 h-5" />;
      case 'failed': return <AlertCircle className="text-red-500 w-5 h-5" />;
      default: return <Clock className="text-gray-400 w-5 h-5" />;
    }
  };

  return (
    <div className="flex flex-col h-screen bg-cyber-bg text-white font-sans">
      {/* Header */}
      <header className="flex items-center justify-between px-8 py-4 bg-cyber-surface border-b border-primary/20 shadow-lg shadow-primary/5">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-primary/10 rounded-lg border border-primary/30">
            <Package className="text-primary w-8 h-8" />
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-tighter">AI TRACKING <span className="text-primary">CORE</span></h1>
            <p className="text-[10px] text-gray-500 uppercase tracking-widest">Intelligent Delivery v1.0</p>
          </div>
        </div>
        
        <div className="flex items-center gap-6">
          <div className="hidden md:flex bg-black/40 border border-white/10 rounded-full px-4 py-1.5 items-center gap-2">
            <Search className="w-4 h-4 text-gray-400" />
            <input 
              type="text" 
              placeholder="Search Package ID..." 
              className="bg-transparent border-none outline-none text-sm w-48 placeholder:text-gray-600"
            />
          </div>
          <Bell className="w-5 h-5 text-gray-400 hover:text-primary cursor-pointer transition-colors" />
          <div className="h-8 w-px bg-white/10 mx-2"></div>
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-full bg-primary/20 flex items-center justify-center border border-primary/40">
              <User className="w-4 h-4 text-primary" />
            </div>
            <span className="text-sm font-medium hidden sm:inline">Admin User</span>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex flex-1 overflow-hidden p-6 gap-6">
        {/* Left Side: Package List */}
        <section className="w-full md:w-1/3 flex flex-col gap-4 overflow-y-auto pr-2">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold flex items-center gap-2">
              <Menu className="w-5 h-5 text-primary" /> Active Shipments
            </h2>
            <span className="bg-primary/10 text-primary text-[10px] px-2 py-1 rounded border border-primary/20 font-bold">
              {packages.length} TOTAL
            </span>
          </div>
          
          <div className="space-y-3">
            {packages.map((pkg) => (
              <div 
                key={pkg.id}
                onClick={() => setSelectedPackage(pkg)}
                className={`p-4 rounded-xl border transition-all cursor-pointer group ${
                  selectedPackage?.id === pkg.id 
                  ? 'bg-primary/5 border-primary shadow-[0_0_15px_rgba(200,255,0,0.1)]' 
                  : 'bg-cyber-surface border-white/5 hover:border-white/20'
                }`}
              >
                <div className="flex justify-between items-start mb-2">
                  <span className="text-xs text-gray-500 font-mono">#{pkg.order_id}</span>
                  {getStatusIcon(pkg.status)}
                </div>
                <h3 className="font-bold text-sm mb-1 group-hover:text-primary transition-colors">
                  São Paulo Hub → Residential Area
                </h3>
                <div className="flex items-center gap-3 mt-3 text-[11px] text-gray-400">
                  <span className="flex items-center gap-1"><MapPin className="w-3 h-3" /> Zone A4</span>
                  <span className="flex items-center gap-1"><Clock className="w-3 h-3" /> ETA: {new Date(pkg.eta).toLocaleTimeString()}</span>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* Right Side: Map and Details */}
        <section className="hidden md:flex flex-1 flex-col gap-6">
          {/* Map View */}
          <div className="flex-1 bg-black rounded-2xl border border-white/5 relative overflow-hidden group">
            {/* Simulated Map Grid */}
            <div className="absolute inset-0 opacity-20 pointer-events-none" style={{
              backgroundImage: 'radial-gradient(circle, #333 1px, transparent 1px)',
              backgroundSize: '24px 24px'
            }}></div>
            
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="text-center">
                <MapPin className="w-12 h-12 text-primary mx-auto mb-4 animate-bounce" />
                <h3 className="text-xl font-bold tracking-tight">LIVE TRACKING ACTIVE</h3>
                <p className="text-sm text-gray-500 max-w-xs mx-auto mt-2">
                  Geospatial data processed via AI-Optimized Routing Engine
                </p>
              </div>
            </div>
            
            {/* Map HUD Components */}
            <div className="absolute top-4 left-4 p-3 bg-black/80 backdrop-blur-md rounded-lg border border-white/10 text-[10px] space-y-1 z-10">
              <div className="flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-primary animate-pulse"></span>
                <span>SYSTEM STATUS: NORMAL</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-blue-500"></span>
                <span>GPS ACCURACY: 2.4m</span>
              </div>
            </div>

            {/* Live Notifications Overlay */}
            <div className="absolute top-4 right-4 flex flex-col gap-2 max-w-[250px] z-10">
              {notifications.map((n, i) => (
                <div key={i} className="bg-primary/20 backdrop-blur-xl border border-primary/40 rounded px-3 py-2 text-[10px] animate-in slide-in-from-right-4">
                  <p className="text-primary font-bold uppercase mb-1">AI Detection Alert</p>
                  <p className="text-white/80">Object: <span className="text-white font-mono">{n.object}</span></p>
                  <p className="text-white/60">Conf: {(n.confidence * 100).toFixed(1)}%</p>
                </div>
              ))}
            </div>

            <div className="absolute bottom-4 right-4 flex gap-2">
              <button className="px-3 py-1.5 bg-primary text-black text-xs font-bold rounded-md hover:scale-105 transition-transform">
                CENTER CAMERA
              </button>
            </div>
          </div>

          {/* Detailed Info (Horizontal Panel) */}
          {selectedPackage && (
            <div className="h-48 bg-cyber-surface rounded-2xl border border-primary/30 p-6 flex gap-10 animate-in fade-in slide-in-from-bottom-4 duration-500">
              <div className="flex-1">
                <div className="flex items-center gap-2 mb-4">
                  <div className="p-1.5 bg-primary/10 rounded border border-primary/20">
                    <Truck className="w-4 h-4 text-primary" />
                  </div>
                  <h4 className="font-bold text-sm uppercase tracking-wider">Package Analysis</h4>
                </div>
                <div className="grid grid-cols-2 gap-y-3 gap-x-6 text-sm">
                  <div>
                    <p className="text-[10px] text-gray-500 uppercase">Weight</p>
                    <p className="font-mono">{selectedPackage.weight} kg</p>
                  </div>
                  <div>
                    <p className="text-[10px] text-gray-500 uppercase">Integrity Score</p>
                    <p className="text-green-400">98% Validated</p>
                  </div>
                  <div>
                    <p className="text-[10px] text-gray-500 uppercase">Current Velocity</p>
                    <p className="font-mono">42.5 km/h</p>
                  </div>
                  <div>
                    <p className="text-[10px] text-gray-500 uppercase">Last Sync</p>
                    <p className="text-xs">45 seconds ago</p>
                  </div>
                </div>
              </div>
              
              <div className="w-px bg-white/5 h-full"></div>
              
              <div className="flex-1">
                <h4 className="font-bold text-sm uppercase tracking-wider mb-4">Visual Evidence</h4>
                <div className="w-full h-24 bg-black/40 rounded border border-white/5 flex items-center justify-center italic text-xs text-gray-600">
                  Waiting for YOLOv8 Stream...
                </div>
              </div>
            </div>
          )}
        </section>
      </main>
    </div>
  );
}

export default App;
