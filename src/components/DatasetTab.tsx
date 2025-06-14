import React, { useState } from 'react';
import { Upload, FileText, BarChart3, CheckCircle, AlertCircle } from 'lucide-react';

export const DatasetTab: React.FC = () => {
  const [uploadedFiles, setUploadedFiles] = useState([
    { name: 'training_data.txt', size: '2.4 MB', status: 'processed' },
    { name: 'validation_data.txt', size: '0.8 MB', status: 'processing' },
  ]);

  const [datasetStats] = useState({
    totalSamples: 15420,
    avgTokenLength: 142,
    vocabularySize: 8950,
    dataQuality: 94,
  });

  return (
    <div className="space-y-8">
      {/* Upload Section */}
      <div className="bg-white rounded-xl shadow-md p-8">
        <h3 className="text-2xl font-bold text-gray-900 mb-6">Dataset Upload</h3>
        <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-blue-400 transition-colors">
          <Upload className="h-12 w-12 text-gray-400 mx-auto mb-4" />
          <h4 className="text-lg font-semibold text-gray-900 mb-2">Upload Training Data</h4>
          <p className="text-gray-600 mb-4">
            Drag and drop your text files here, or click to browse
          </p>
          <button className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 transition-colors">
            Choose Files
          </button>
          <p className="text-sm text-gray-500 mt-2">
            Supported formats: .txt, .csv, .json (Max 100MB)
          </p>
        </div>
      </div>

      {/* File Management */}
      <div className="bg-white rounded-xl shadow-md p-8">
        <h3 className="text-2xl font-bold text-gray-900 mb-6">Uploaded Files</h3>
        <div className="space-y-4">
          {uploadedFiles.map((file, index) => (
            <div key={index} className="flex items-center justify-between p-4 border rounded-lg">
              <div className="flex items-center space-x-3">
                <FileText className="h-8 w-8 text-blue-600" />
                <div>
                  <h4 className="font-medium text-gray-900">{file.name}</h4>
                  <p className="text-sm text-gray-500">{file.size}</p>
                </div>
              </div>
              <div className="flex items-center space-x-3">
                {file.status === 'processed' ? (
                  <span className="flex items-center text-green-600">
                    <CheckCircle className="h-4 w-4 mr-1" />
                    Processed
                  </span>
                ) : (
                  <span className="flex items-center text-yellow-600">
                    <AlertCircle className="h-4 w-4 mr-1" />
                    Processing
                  </span>
                )}
                <button className="text-blue-600 hover:text-blue-800">View</button>
                <button className="text-red-600 hover:text-red-800">Remove</button>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Dataset Statistics */}
      <div className="bg-white rounded-xl shadow-md p-8">
        <h3 className="text-2xl font-bold text-gray-900 mb-6">Dataset Statistics</h3>
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
          <div className="text-center p-6 bg-blue-50 rounded-lg">
            <BarChart3 className="h-8 w-8 text-blue-600 mx-auto mb-2" />
            <h4 className="text-2xl font-bold text-blue-900">{datasetStats.totalSamples.toLocaleString()}</h4>
            <p className="text-blue-700">Total Samples</p>
          </div>
          <div className="text-center p-6 bg-green-50 rounded-lg">
            <FileText className="h-8 w-8 text-green-600 mx-auto mb-2" />
            <h4 className="text-2xl font-bold text-green-900">{datasetStats.avgTokenLength}</h4>
            <p className="text-green-700">Avg Token Length</p>
          </div>
          <div className="text-center p-6 bg-purple-50 rounded-lg">
            <BarChart3 className="h-8 w-8 text-purple-600 mx-auto mb-2" />
            <h4 className="text-2xl font-bold text-purple-900">{datasetStats.vocabularySize.toLocaleString()}</h4>
            <p className="text-purple-700">Vocabulary Size</p>
          </div>
          <div className="text-center p-6 bg-orange-50 rounded-lg">
            <CheckCircle className="h-8 w-8 text-orange-600 mx-auto mb-2" />
            <h4 className="text-2xl font-bold text-orange-900">{datasetStats.dataQuality}%</h4>
            <p className="text-orange-700">Data Quality</p>
          </div>
        </div>
      </div>

      {/* Data Preprocessing */}
      <div className="bg-white rounded-xl shadow-md p-8">
        <h3 className="text-2xl font-bold text-gray-900 mb-6">Preprocessing Options</h3>
        <div className="grid md:grid-cols-2 gap-8">
          <div>
            <h4 className="font-semibold text-gray-900 mb-4">Text Cleaning</h4>
            <div className="space-y-3">
              <label className="flex items-center">
                <input type="checkbox" className="mr-3" defaultChecked />
                <span className="text-gray-700">Remove HTML tags</span>
              </label>
              <label className="flex items-center">
                <input type="checkbox" className="mr-3" defaultChecked />
                <span className="text-gray-700">Normalize whitespace</span>
              </label>
              <label className="flex items-center">
                <input type="checkbox" className="mr-3" />
                <span className="text-gray-700">Convert to lowercase</span>
              </label>
              <label className="flex items-center">
                <input type="checkbox" className="mr-3" />
                <span className="text-gray-700">Remove special characters</span>
              </label>
            </div>
          </div>
          <div>
            <h4 className="font-semibold text-gray-900 mb-4">Tokenization</h4>
            <div className="space-y-3">
              <label className="flex items-center">
                <input type="radio" name="tokenizer" className="mr-3" defaultChecked />
                <span className="text-gray-700">GPT-2 Tokenizer</span>
              </label>
              <label className="flex items-center">
                <input type="radio" name="tokenizer" className="mr-3" />
                <span className="text-gray-700">Custom Tokenizer</span>
              </label>
              <div className="mt-4">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Max Sequence Length
                </label>
                <input 
                  type="number" 
                  className="w-full p-2 border rounded-lg" 
                  defaultValue={512}
                />
              </div>
            </div>
          </div>
        </div>
        <div className="mt-6 pt-6 border-t">
          <button className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 transition-colors">
            Apply Preprocessing
          </button>
        </div>
      </div>
    </div>
  );
};