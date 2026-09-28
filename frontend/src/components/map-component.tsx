"use client";

import { useEffect, useState } from "react";
import { MapContainer, TileLayer, Marker, Popup, CircleMarker } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import L from "leaflet";
export interface IntelligenceObservation {
  id: string;
  farm_id: string;
  timestamp: string;
  crop: string;
  disease: string;
  isHealthy: boolean;
  confidence: number;
  latitude: number;
  longitude: number;
  region: string;
  temperature: number;
  humidity: number;
}

// Fix for default marker icons in Leaflet with Next.js
delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png",
  iconUrl: "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png",
  shadowUrl: "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png",
});

interface MapProps {
  data: IntelligenceObservation[];
}

export default function IntelligenceMap({ data }: MapProps) {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) return <div className="h-[400px] w-full bg-slate-100 animate-pulse rounded-md" />;

  // Center around India (default focus)
  const defaultCenter: [number, number] = [20.5937, 78.9629];

  return (
    <div className="h-[400px] w-full rounded-md overflow-hidden border border-slate-200">
      <MapContainer 
        center={defaultCenter} 
        zoom={4} 
        style={{ height: "100%", width: "100%" }}
        scrollWheelZoom={false}
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        {data.map((obs, idx) => {
          if (!obs.latitude || !obs.longitude) return null;
          
          const isHealthy = obs.isHealthy || obs.disease.toLowerCase().includes("healthy");
          const color = isHealthy ? "#22c55e" : (obs.confidence > 0.8 ? "#ef4444" : "#eab308");

          return (
            <CircleMarker
              key={idx}
              center={[obs.latitude, obs.longitude]}
              pathOptions={{ color, fillColor: color, fillOpacity: 0.7 }}
              radius={isHealthy ? 4 : Math.max(4, obs.severity * 15)}
            >
              <Popup>
                <div className="p-1">
                  <p className="font-bold text-sm mb-1">{obs.crop} - {obs.disease}</p>
                  <p className="text-xs text-slate-500">Date: {new Date(obs.timestamp).toLocaleDateString()}</p>
                  <p className="text-xs text-slate-500">Severity: {(obs.severity * 100).toFixed(0)}%</p>
                  <p className="text-xs text-slate-500">Region: {obs.region}</p>
                </div>
              </Popup>
            </CircleMarker>
          );
        })}
      </MapContainer>
    </div>
  );
}
