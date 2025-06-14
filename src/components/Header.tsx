import React from 'react';
import { Brain, Zap } from 'lucide-react';

export const Header: React.FC = () => {
  return (
    <header className="bg-gradient-to-r from-blue-900 via-purple-900 to-indigo-900 text-white shadow-lg">
      <div className="container mx-auto px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="bg-white/10 p-2 rounded-lg backdrop-blur-sm">
              <Brain className="h-8 w-8 text-blue-300" />
            </div>
            <div>
              <h1 className="text-2xl font-bold">GPT-2 Fine-tuning Lab</h1>
              <p className="text-blue-200 text-sm">Generative AI Internship Project</p>
            </div>
          </div>
          <div className="flex items-center space-x-2 bg-white/10 px-4 py-2 rounded-full backdrop-blur-sm">
            <Zap className="h-4 w-4 text-yellow-300" />
            <span className="text-sm font-medium">Training Ready</span>
          </div>
        </div>
      </div>
    </header>
  );
};