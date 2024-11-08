import { create } from 'zustand';

import { ImageStore } from '@features/train/types/analyze.type'

export const useImageStore = create<ImageStore>((set) => ({
   uploadedImage: null,
   setUploadedImage: (image) => set({ uploadedImage: image }),
}));