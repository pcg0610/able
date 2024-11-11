export interface FeatureMapProps {
  projectName: string;
  resultName: string;
  epochName: string;
  blockIds: string;
}

export interface EpochResponse {
  epochs: string[];
  hasNext: boolean;
}

export interface FeatureMapResponse {
  image: string;
  classScores: ClassScore[];
}

export interface CreateFeatureMapProps {
  projectName: string;
  resultName: string;
  epochName: string;
  deviceIndex: number;
  image: string | null;
}

export interface HeatMapResponse {
  originalImg: string;
  heatmapImg: string;
  classScores: ClassScore[];
}

export interface ClassScore {
  className: string;
  classScore: number;
}

export interface ImageStore {
  uploadedImage: string | null;
  heatmapImage: string | null;
  classScores: ClassScore[];
  heatMapId: string;

  setUploadedImage: (image: string | null) => void;
  setHeatMapId: (id: string) => void;
  setHeatMapImage: (data: { heatmapImage: string; classScores: ClassScore[] }) => void;
  setAllImage: (data: { uploadedImage: string; heatmapImage: string; classScores: ClassScore[] }) => void;

  resetImage: () => void;
}
