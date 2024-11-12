import { create } from 'zustand';

import { ImageStore } from '@features/train/types/analyze.type'


export const useImageStore = create<ImageStore>((set) => ({
   uploadedImage: null,
   heatmapImage: null,
   classScores: [],

   heatMapId: '',

   setUploadedImage: (image) => set({ uploadedImage: image }),
   setHeatMapId: (id) => set({ heatMapId: id }),

   setHeatMapImage: (data) => {
      set({
         heatmapImage: data.heatmapImage,
         classScores: data.classScores,
      });
   },

   setAllImage: (data) => {
      set({
         uploadedImage: data.uploadedImage,
         heatmapImage: data.heatmapImage,
         classScores: data.classScores,
      });
   },

   resetImage: () => {
      set({
         uploadedImage: null,
         heatmapImage: null,
         classScores: [],
         heatMapId: '',
      });
   },

}));