"use client";

import { useState, useEffect, useCallback } from "react";
import { Cloud, Droplets, MapPin, Sun, Wind, Thermometer, AlertCircle, Loader2, Search, Navigation } from "lucide-react";
import { useTranslation } from "@/lib/i18n";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

interface LiveWeather {
  temperature: number;
  humidity: number;
  windspeed: number;
  locationLabel: string;
  cityName?: string;
  timestamp: string;
}

const POPULAR_LOCATIONS = [
  { name: "Hyderabad", lat: 17.3850, lon: 78.4867 },
  { name: "Warangal", lat: 17.9689, lon: 79.5941 },
  { name: "Karimnagar", lat: 18.4386, lon: 79.1288 },
  { name: "Nizamabad", lat: 18.6725, lon: 78.0941 },
  { name: "Guntur", lat: 16.3067, lon: 80.4365 },
  { name: "Vijayawada", lat: 16.5062, lon: 80.6480 },
  { name: "Bengaluru", lat: 12.9716, lon: 77.5946 },
  { name: "Delhi", lat: 28.6139, lon: 77.2090 },
];

export default function WeatherPage() {
  const { t } = useTranslation();
  const [weather, setWeather] = useState<LiveWeather | null>(null);
  const [locationStatus, setLocationStatus] = useState<"loading" | "unavailable" | "loaded">("loading");
  const [error, setError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [isSearching, setIsSearching] = useState(false);

  const getLabel = (key: string, fallback: string) => {
    const val = t(key);
    return !val || val === key ? fallback : val;
  };

  const fetchWeatherForCoords = useCallback(async (latitude: number, longitude: number, cityName?: string) => {
    setLocationStatus("loading");
    setError(null);
    try {
      const res = await fetch(
        `https://api.open-meteo.com/v1/forecast?latitude=${latitude}&longitude=${longitude}&current=temperature_2m,relative_humidity_2m,wind_speed_10m&timezone=auto`
      );
      if (!res.ok) throw new Error("Weather API error");
      const data = await res.json();
      const current = data.current;

      let displayCity = cityName;
      if (!displayCity) {
        // Try reverse geocoding to find city name
        try {
          const geoRes = await fetch(
            `https://nominatim.openstreetmap.org/reverse?format=json&lat=${latitude}&lon=${longitude}&zoom=10`
          );
          if (geoRes.ok) {
            const geoData = await geoRes.json();
            displayCity = geoData.address?.city || geoData.address?.town || geoData.address?.county || geoData.address?.state;
          }
        } catch {
          // Ignore reverse geocode failures
        }
      }

      setWeather({
        temperature: current.temperature_2m,
        humidity: current.relative_humidity_2m,
        windspeed: current.wind_speed_10m,
        locationLabel: `${latitude.toFixed(2)}°N, ${longitude.toFixed(2)}°E`,
        cityName: displayCity,
        timestamp: new Date(current.time).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
      });
      setLocationStatus("loaded");
    } catch {
      setError("Could not fetch live weather data.");
      setLocationStatus("unavailable");
    }
  }, []);

  const handleUseGPS = useCallback(() => {
    if (!("geolocation" in navigator)) {
      setLocationStatus("unavailable");
      setError("Geolocation is not supported by your browser.");
      return;
    }

    setLocationStatus("loading");
    navigator.geolocation.getCurrentPosition(
      (position) => {
        const { latitude, longitude } = position.coords;
        fetchWeatherForCoords(latitude, longitude);
      },
      () => {
        // Fallback to default agricultural region (Hyderabad) if browser blocks geolocation
        fetchWeatherForCoords(17.3850, 78.4867, "Hyderabad (Default)");
      },
      { timeout: 8000 }
    );
  }, [fetchWeatherForCoords]);

  const handleCitySearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!searchQuery.trim()) return;

    setIsSearching(true);
    try {
      const res = await fetch(
        `https://geocoding-api.open-meteo.com/v1/search?name=${encodeURIComponent(searchQuery)}&count=1&language=en&format=json`
      );
      if (!res.ok) throw new Error("Search failed");
      const data = await res.json();
      if (data.results && data.results.length > 0) {
        const place = data.results[0];
        await fetchWeatherForCoords(place.latitude, place.longitude, `${place.name}, ${place.admin1 || place.country}`);
        setSearchQuery("");
      } else {
        setError(`Location "${searchQuery}" not found. Please try another city.`);
      }
    } catch {
      setError("Failed to search location. Please check your connection.");
    } finally {
      setIsSearching(false);
    }
  };

  useEffect(() => {
    handleUseGPS();
  }, [handleUseGPS]);

  return (
    <div className="flex flex-col gap-6 max-w-3xl mx-auto w-full">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-3xl sm:text-4xl font-extrabold tracking-tight">
            {getLabel("weather.title", "Local Weather")}
          </h1>
          <p className="text-muted-foreground mt-2 text-base">
            {getLabel("weather.subtitle", "Real-time microclimate conditions for your crop area.")}
          </p>
        </div>
        {locationStatus === "loaded" && weather && (
          <div className="flex items-center gap-2 text-sm font-medium text-muted-foreground bg-muted/60 px-4 py-2 rounded-full w-fit border">
            <MapPin className="h-4 w-4 text-primary shrink-0" />
            <span>{weather.cityName ? `${weather.cityName} · ${weather.locationLabel}` : weather.locationLabel}</span>
          </div>
        )}
      </div>

      {/* Location Search Bar & Controls */}
      <div className="flex flex-col gap-3 p-4 rounded-2xl border bg-card/60 shadow-sm">
        <form onSubmit={handleCitySearch} className="flex gap-2">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-2.5 h-4 w-4 text-muted-foreground" />
            <Input
              type="text"
              placeholder="Search your city or district (e.g. Warangal, Guntur, Hyderabad)..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-9 rounded-xl"
            />
          </div>
          <Button type="submit" disabled={isSearching} className="rounded-xl shrink-0">
            {isSearching ? <Loader2 className="h-4 w-4 animate-spin" /> : "Search"}
          </Button>
          <Button 
            type="button" 
            variant="outline" 
            onClick={handleUseGPS} 
            title="Auto-detect using GPS"
            className="rounded-xl shrink-0 gap-1.5"
          >
            <Navigation className="h-4 w-4 text-primary" />
            <span className="hidden sm:inline">Use GPS</span>
          </Button>
        </form>

        {/* Quick Location Pills */}
        <div className="flex flex-wrap items-center gap-1.5 text-xs">
          <span className="text-muted-foreground font-medium mr-1">Quick Select:</span>
          {POPULAR_LOCATIONS.map((loc) => (
            <button
              key={loc.name}
              type="button"
              onClick={() => fetchWeatherForCoords(loc.lat, loc.lon, loc.name)}
              className="px-2.5 py-1 rounded-lg bg-muted hover:bg-muted/80 text-foreground transition-colors font-medium border border-border/40"
            >
              {loc.name}
            </button>
          ))}
        </div>
      </div>

      {/* States */}
      {locationStatus === "loading" && (
        <div className="flex flex-col items-center justify-center gap-3 py-16 text-muted-foreground">
          <Loader2 className="h-8 w-8 animate-spin text-primary" />
          <p className="text-sm font-medium">Fetching real-time weather from Open-Meteo…</p>
        </div>
      )}

      {locationStatus === "unavailable" && (
        <div className="flex flex-col items-center justify-center gap-4 py-16 text-center border rounded-2xl border-border/50 bg-muted/20">
          <AlertCircle className="h-10 w-10 text-amber-500/80" />
          <div>
            <p className="font-semibold text-foreground">
              {error ?? "Location access is required for local weather."}
            </p>
            <p className="text-sm text-muted-foreground mt-1">
              Select one of the cities above or search your town to load local weather.
            </p>
          </div>
        </div>
      )}

      {locationStatus === "loaded" && weather && (
        <>
          {/* Current Conditions Card */}
          <div className="bg-gradient-to-br from-blue-600 to-emerald-700 text-white rounded-3xl shadow-lg p-6 sm:p-8">
            <div className="flex justify-between items-center text-blue-100 text-sm font-medium mb-4">
              <span>{getLabel("weather.current", "Current Conditions")} · {weather.timestamp}</span>
              {weather.cityName && <span className="bg-white/15 px-3 py-0.5 rounded-full text-xs font-semibold">{weather.cityName}</span>}
            </div>
            <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-6">
              <div className="flex items-center gap-6">
                <Sun className="h-16 w-16 text-yellow-300 drop-shadow-md shrink-0" />
                <div>
                  <div className="text-5xl font-extrabold tracking-tighter">{weather.temperature}°C</div>
                  <p className="text-sm text-blue-100 mt-1">Live data · Open-Meteo</p>
                </div>
              </div>
              <div className="flex gap-8">
                {[
                  { icon: Droplets, value: `${weather.humidity}%`, label: getLabel("weather.humidity", "Humidity") },
                  { icon: Wind, value: `${weather.windspeed} km/h`, label: getLabel("weather.wind", "Wind Speed") },
                  { icon: Thermometer, value: `${weather.temperature}°C`, label: getLabel("weather.temperature", "Temperature") },
                ].map(({ icon: Icon, value, label }) => (
                  <div key={label} className="flex flex-col items-center gap-1.5">
                    <Icon className="h-5 w-5 text-blue-200" />
                    <span className="text-sm font-bold">{value}</span>
                    <span className="text-xs text-blue-100 font-medium">{label}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Agronomic Context Insight */}
          <div className="p-5 rounded-2xl border bg-card/80 shadow-sm flex flex-col gap-2">
            <h3 className="font-semibold text-sm text-foreground flex items-center gap-2">
              <Thermometer className="h-4 w-4 text-primary" />
              Agronomic Microclimate Advisory
            </h3>
            <p className="text-xs text-muted-foreground leading-relaxed">
              At {weather.temperature}°C and {weather.humidity}% relative humidity, 
              {weather.humidity > 80 
                ? " high humidity increases risk for fungal blights (e.g. Late Blight, Blast). Monitor lower leaf canopies closely."
                : weather.humidity < 45 
                ? " dry conditions favor spider mite proliferation and moisture stress. Ensure adequate soil irrigation."
                : " moderate conditions are favorable for crop vegetative growth. Continue regular field scouting."}
            </p>
          </div>
        </>
      )}
    </div>
  );
}
