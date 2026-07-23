"use client";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Cloud, CloudRain, CloudSun, Droplets, MapPin, Sun, Wind } from "lucide-react";

const forecastData = [
  { day: "Today", temp: "28°C", condition: "Sunny", icon: Sun, color: "text-orange-500" },
  { day: "Tomorrow", temp: "26°C", condition: "Partly Cloudy", icon: CloudSun, color: "text-blue-400" },
  { day: "Wednesday", temp: "24°C", condition: "Rain", icon: CloudRain, color: "text-blue-500" },
  { day: "Thursday", temp: "22°C", condition: "Heavy Rain", icon: CloudRain, color: "text-blue-600" },
  { day: "Friday", temp: "25°C", condition: "Cloudy", icon: Cloud, color: "text-gray-500" },
];

export default function WeatherPage() {
  return (
    <div className="flex flex-col gap-6 max-w-5xl mx-auto">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Weather Forecast</h1>
          <p className="text-muted-foreground mt-2">Hyper-local weather data for your registered farm locations.</p>
        </div>
        <div className="flex items-center gap-2 text-sm font-medium text-muted-foreground bg-muted/50 px-4 py-2 rounded-full w-fit">
          <MapPin className="h-4 w-4" />
          <span>Central Valley Farm, CA</span>
        </div>
      </div>

      <div className="grid gap-6 md:grid-cols-3">
        <Card className="md:col-span-2 bg-gradient-to-br from-blue-500 to-blue-700 text-white border-none shadow-md overflow-hidden relative">
          <div className="absolute top-0 right-0 p-8 opacity-20 pointer-events-none">
            <Sun className="h-48 w-48 animate-pulse duration-10000" />
          </div>
          <CardHeader>
            <CardTitle className="text-blue-100">Current Conditions</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex flex-col md:flex-row items-center justify-between z-10 relative">
              <div className="flex items-center gap-6">
                <Sun className="h-24 w-24 text-yellow-300 drop-shadow-md" />
                <div>
                  <div className="text-6xl font-bold tracking-tighter">28°C</div>
                  <p className="text-xl font-medium text-blue-100 mt-1">Clear & Sunny</p>
                  <p className="text-sm text-blue-200">Feels like 30°C</p>
                </div>
              </div>
              <div className="flex gap-8 mt-6 md:mt-0">
                <div className="flex flex-col items-center gap-2">
                  <Droplets className="h-6 w-6 text-blue-200" />
                  <span className="text-sm font-medium">45%</span>
                  <span className="text-xs text-blue-200">Humidity</span>
                </div>
                <div className="flex flex-col items-center gap-2">
                  <Wind className="h-6 w-6 text-blue-200" />
                  <span className="text-sm font-medium">12 km/h</span>
                  <span className="text-xs text-blue-200">Wind</span>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Agri-Insights</CardTitle>
            <CardDescription>Weather impact on crops</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="bg-green-500/10 border border-green-500/20 p-4 rounded-lg">
              <h4 className="font-semibold text-green-700 dark:text-green-400 text-sm mb-1">Favorable Conditions</h4>
              <p className="text-xs text-muted-foreground">Current weather is ideal for planned pesticide application. Wind speeds are well below the threshold.</p>
            </div>
            <div className="bg-yellow-500/10 border border-yellow-500/20 p-4 rounded-lg">
              <h4 className="font-semibold text-yellow-700 dark:text-yellow-400 text-sm mb-1">Upcoming Rain</h4>
              <p className="text-xs text-muted-foreground">Heavy rain expected on Thursday. Postpone any irrigation schedules.</p>
            </div>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>5-Day Forecast</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
            {forecastData.map((day, index) => (
              <div key={index} className="flex flex-col items-center justify-center p-4 rounded-lg border bg-card hover:bg-muted/50 transition-colors">
                <span className="text-sm font-medium text-muted-foreground mb-4">{day.day}</span>
                <day.icon className={`h-10 w-10 mb-4 ${day.color}`} />
                <span className="text-2xl font-bold mb-1">{day.temp}</span>
                <span className="text-xs font-medium text-center">{day.condition}</span>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
