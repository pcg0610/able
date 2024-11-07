export interface FeatureMapProps {
  projectName: string;
  resultName: string;
  epochName: string;
  blockIds: string[];
}

export interface FeatureMapResponse {
  blockId: string;
  img: string;
}

export interface EpochResponse {
  epochs: string[];
  hasNext: boolean;
}

export interface ImageStore {
  uploadedImage: string | null;
  setUploadedImage: (image: string | null) => void;
}
