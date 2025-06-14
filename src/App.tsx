import React, { useState } from 'react';
import { Header } from './components/Header';
import { TabNavigation } from './components/TabNavigation';
import { OverviewTab } from './components/OverviewTab';
import { DatasetTab } from './components/DatasetTab';
import { TrainingTab } from './components/TrainingTab';
import { PlaygroundTab } from './components/PlaygroundTab';
import { EvaluationTab } from './components/EvaluationTab';
import { ResourcesTab } from './components/ResourcesTab';

function App() {
  const [activeTab, setActiveTab] = useState('overview');

  const renderActiveTab = () => {
    switch (activeTab) {
      case 'overview':
        return <OverviewTab />;
      case 'dataset':
        return <DatasetTab />;
      case 'training':
        return <TrainingTab />;
      case 'playground':
        return <PlaygroundTab />;
      case 'evaluation':
        return <EvaluationTab />;
      case 'resources':
        return <ResourcesTab />;
      default:
        return <OverviewTab />;
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <Header />
      <TabNavigation activeTab={activeTab} setActiveTab={setActiveTab} />
      <main className="container mx-auto px-6 py-8">
        {renderActiveTab()}
      </main>
    </div>
  );
}

export default App;