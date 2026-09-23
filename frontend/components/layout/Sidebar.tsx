"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { 
  LayoutDashboard, 
  Pill, 
  Network, 
  BookOpen, 
  Activity 
} from "lucide-react";

export function Sidebar() {
  const pathname = usePathname();

  const navItems = [
    { label: "Overview", href: "/", icon: LayoutDashboard },
    { label: "Drug Explorer", href: "/drugs", icon: Pill },
    { label: "Scenario Lab", href: "/scenarios", icon: Network },
    { label: "Methodology", href: "/methodology", icon: BookOpen },
  ];

  return (
    <aside className="w-64 bg-slate-900 border-r border-slate-800 flex flex-col justify-between shrink-0 hidden md:flex">
      <div>
        <div className="p-5 border-b border-slate-800 flex items-center space-x-3">
          <div className="p-2 bg-sky-500/10 rounded-lg text-sky-400 border border-sky-500/20">
            <Activity className="w-6 h-6" />
          </div>
          <div>
            <h1 className="font-bold text-slate-100 tracking-tight text-lg">MedCascade</h1>
            <p className="text-xs text-slate-400">Shortage & Stress Intelligence</p>
          </div>
        </div>

        <nav className="p-3 space-y-1">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = pathname === item.href || (item.href !== "/" && pathname.startsWith(item.href));

            return (
              <Link
                key={item.href}
                href={item.href}
                className={`flex items-center space-x-3 px-3.5 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                  isActive
                    ? "bg-sky-500/10 text-sky-400 border border-sky-500/20"
                    : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/60"
                }`}
              >
                <Icon className="w-4 h-4 shrink-0" />
                <span>{item.label}</span>
              </Link>
            );
          })}
        </nav>
      </div>

      <div className="p-4 border-t border-slate-800/80">
        <div className="bg-slate-950/60 p-3 rounded-lg border border-slate-800/60 text-xs text-slate-400 space-y-1">
          <div className="font-medium text-slate-300">Phase 9 Dashboard</div>
          <div className="text-[11px] text-slate-500">FastAPI ML + Graph B Engine</div>
        </div>
      </div>
    </aside>
  );
}
