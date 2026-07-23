"use client";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  PieChart, Pie, Cell, LineChart, Line
} from "recharts";
import { motion } from "framer-motion";
import { useTranslation } from "@/lib/i18n";

const diseasesByMonth = [
  { name: "Jan", earlyBlight: 40, lateBlight: 24, healthy: 120 },
  { name: "Feb", earlyBlight: 30, lateBlight: 13, healthy: 130 },
  { name: "Mar", earlyBlight: 20, lateBlight: 38, healthy: 150 },
  { name: "Apr", earlyBlight: 27, lateBlight: 39, healthy: 110 },
  { name: "May", earlyBlight: 18, lateBlight: 48, healthy: 180 },
  { name: "Jun", earlyBlight: 23, lateBlight: 38, healthy: 200 },
];

const cropDistribution = [
  { name: "Tomatoes", value: 400 },
  { name: "Potatoes", value: 300 },
  { name: "Wheat", value: 300 },
  { name: "Corn", value: 200 },
];

const accuracyData = [
  { name: "Week 1", accuracy: 92 },
  { name: "Week 2", accuracy: 94 },
  { name: "Week 3", accuracy: 93 },
  { name: "Week 4", accuracy: 96 },
  { name: "Week 5", accuracy: 95 },
  { name: "Week 6", accuracy: 98 },
];

const COLORS = ["#22c55e", "#eab308", "#3b82f6", "#a855f7"];

export default function AnalyticsPage() {
  const { t } = useTranslation();

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: { opacity: 1, transition: { staggerChildren: 0.1 } },
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 15 },
    visible: { opacity: 1, y: 0, transition: { duration: 0.5 } },
  };

  return (
    <motion.div 
      className="flex flex-col gap-6"
      variants={containerVariants}
      initial="hidden"
      animate="visible"
    >
      <div>
        <h1 className="text-3xl font-bold tracking-tight">{t("nav.analytics")}</h1>
        <p className="text-muted-foreground mt-2">Deep dive into your farm's health metrics and AI performance.</p>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        <motion.div variants={itemVariants}>
          <Card className="hover:shadow-md transition-all h-full">
            <CardHeader>
              <CardTitle>Disease Trends (6 Months)</CardTitle>
              <CardDescription>Monthly detection rates of common diseases.</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-[300px]">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={diseasesByMonth}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} opacity={0.3} />
                    <XAxis dataKey="name" axisLine={false} tickLine={false} />
                    <YAxis axisLine={false} tickLine={false} />
                    <Tooltip cursor={{ fill: 'transparent' }} contentStyle={{ borderRadius: '10px' }} />
                    <Legend />
                    <Bar dataKey="earlyBlight" stackId="a" fill="#ef4444" radius={[0, 0, 4, 4]} />
                    <Bar dataKey="lateBlight" stackId="a" fill="#f97316" />
                    <Bar dataKey="healthy" stackId="a" fill="#22c55e" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        </motion.div>

        <motion.div variants={itemVariants}>
          <Card className="hover:shadow-md transition-all h-full">
            <CardHeader>
              <CardTitle>Crop Distribution</CardTitle>
              <CardDescription>Scanned crop ratio across your fields.</CardDescription>
            </CardHeader>
            <CardContent className="flex justify-center">
              <div className="h-[300px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={cropDistribution}
                      cx="50%"
                      cy="50%"
                      innerRadius={60}
                      outerRadius={100}
                      paddingAngle={5}
                      dataKey="value"
                    >
                      {cropDistribution.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip contentStyle={{ borderRadius: '10px' }} />
                    <Legend />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        </motion.div>

        <motion.div variants={itemVariants} className="md:col-span-2">
          <Card className="hover:shadow-md transition-all">
            <CardHeader>
              <CardTitle>AI Detection Accuracy</CardTitle>
              <CardDescription>Model confidence over time.</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-[300px]">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={accuracyData}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} opacity={0.3} />
                    <XAxis dataKey="name" axisLine={false} tickLine={false} />
                    <YAxis domain={[80, 100]} axisLine={false} tickLine={false} />
                    <Tooltip contentStyle={{ borderRadius: '10px' }} />
                    <Line type="monotone" dataKey="accuracy" stroke="#3b82f6" strokeWidth={3} dot={{ r: 6, fill: "#3b82f6", strokeWidth: 2, stroke: "#fff" }} activeDot={{ r: 8 }} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        </motion.div>
      </div>
    </motion.div>
  );
}
