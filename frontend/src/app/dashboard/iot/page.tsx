"use client";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Droplets, ThermometerSun, Wind, FlaskConical, Activity } from "lucide-react";

const sensorData = [
  {
    id: "soil-moisture",
    name: "Soil Moisture",
    value: "42%",
    status: "Optimal",
    icon: Droplets,
    color: "text-blue-500",
    description: "Current soil water content",
  },
  {
    id: "temperature",
    name: "Temperature",
    value: "24°C",
    status: "Normal",
    icon: ThermometerSun,
    color: "text-orange-500",
    description: "Ambient air temperature",
  },
  {
    id: "humidity",
    name: "Humidity",
    value: "65%",
    status: "High",
    icon: Wind,
    color: "text-teal-500",
    description: "Relative air humidity",
  },
  {
    id: "npk",
    name: "NPK Levels",
    value: "14-10-10",
    status: "Requires Attention",
    icon: FlaskConical,
    color: "text-purple-500",
    description: "Nitrogen, Phosphorus, Potassium",
  },
];

export default function IoTMonitoringPage() {
  return (
    <div className="flex flex-col gap-6 max-w-6xl mx-auto">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">IoT Sensor Data</h1>
        <p className="text-muted-foreground mt-2">Real-time monitoring of your field conditions.</p>
      </div>

      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
        {sensorData.map((sensor) => (
          <Card key={sensor.id} className="relative overflow-hidden group">
            <div className="absolute inset-0 bg-gradient-to-br from-primary/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-sm font-medium">{sensor.name}</CardTitle>
              <sensor.icon className={`h-5 w-5 ${sensor.color}`} />
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold">{sensor.value}</div>
              <div className="flex items-center justify-between mt-2">
                <p className="text-xs text-muted-foreground">{sensor.description}</p>
                <span className={`text-xs font-semibold px-2 py-1 rounded-full ${
                  sensor.status === "Optimal" || sensor.status === "Normal" 
                    ? "bg-green-500/10 text-green-600 dark:text-green-400" 
                    : sensor.status === "High" 
                    ? "bg-yellow-500/10 text-yellow-600 dark:text-yellow-400"
                    : "bg-destructive/10 text-destructive"
                }`}>
                  {sensor.status}
                </span>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      <Card className="mt-4">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Activity className="h-5 w-5 text-primary" />
            Live Monitoring Feed
          </CardTitle>
          <CardDescription>Continuous data stream from your field sensors over the last 24 hours.</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-[350px] w-full border border-dashed rounded-lg flex flex-col items-center justify-center bg-muted/20 text-muted-foreground">
            <Activity className="h-12 w-12 mb-4 opacity-20 animate-pulse" />
            <p>Chart Visualization (Trend Data)</p>
            <p className="text-xs mt-2">Historical trends will appear here when connected to the data stream.</p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
