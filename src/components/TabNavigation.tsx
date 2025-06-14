import React from 'react';
import { BookOpen, Database, Settings, Play, TrendingUp, Lightbulb } from 'lucide-react';

interface TabNavigationProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
}

const tabs = [
  { id: 'overview', label: 'Overview', icon: BookOpen },
  { id: 'dataset', label: 'Dataset', icon: Database },
  { id: 'training', label: 'Training', icon: Settings },
  { id: 'playground', label: 'Playground', icon: Play },
  { id: 'evaluation', label: 'Evaluation', icon: TrendingUp },
  { id: 'resources', label: 'Resources', icon: Lightbulb },
];

export const TabNavigation: React.FC<TabNavigationProps> = ({ activeTab, setActiveTab }) => {
  return (
    <nav className="bg-white shadow-sm border-b sticky top-0 z-10">
      <div className="container mx-auto px-6">
        <div className="flex space-x-0 overflow-x-auto">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center space-x-2 px-6 py-4 border-b-2 transition-all duration-200 whitespace-nowrap ${
                  activeTab === tab.id
                    ? 'border-blue-500 text-blue-600 bg-blue-50'
                    : 'border-transparent text-gray-600 hover:text-gray-900 hover:bg-gray-50'
                }`}
              >
                <Icon className="h-4 w-4" />
                <span className="font-medium">{tab.label}</span>
              </button>
            );
          })}
        </div>
      </div>
    </nav>
  );
};