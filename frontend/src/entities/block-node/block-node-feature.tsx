import { Handle, Position } from '@xyflow/react';
import React, { useMemo } from 'react';

import * as S from '@entities/block-node/block-node.style';
import Common from '@shared/styles/common';
import { blockColors } from '@shared/constants/block';
import { capitalizeFirstLetter } from '@shared/utils/formatters.util';
import { BlockItem } from '@/features/canvas/types/block.type';
import { useImageStore } from '@entities/train/model/train.model';

import UploadImageIcon from '@icons/uploadImage.svg?react';

interface BlockNodeFeatureProps {
   data: {
      block: BlockItem;
      onFieldChange: (fieldName: string, value: string) => void;
      featureMap?: Array<{ blockId: string; img: string }>;
   };
   sourcePosition?: Position;
   targetPosition?: Position;
}

const BlockNodeFeature = ({
   data,
   sourcePosition = Position.Bottom,
   targetPosition = Position.Top,
}: BlockNodeFeatureProps) => {
   const { uploadedImage, setUploadedImage } = useImageStore();

   const blockColor = useMemo(
      () => (data?.block?.type ? blockColors[data.block.type] : Common.colors.gray200),
      [data?.block?.type]
   );

   const blockImage = useMemo(() => {
      const feature = data.featureMap?.find((item) => item.blockId === data.block.id);
      return feature?.img || null;
   }, [data.featureMap, data.block.id]);

   const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
      if (event.target.files && event.target.files[0]) {
         const file = event.target.files[0];
         const reader = new FileReader();
         reader.onload = () => {
            setUploadedImage(reader.result as string);
         };
         reader.readAsDataURL(file);
      }
   };

   const handleClickUpload = () => {
      document.getElementById('fileUpload')?.click();
   };

   return (
      <S.Container blockColor={blockColor}>
         <Handle type="target" position={targetPosition} />
         <S.Label>{capitalizeFirstLetter(data?.block?.name || 'Unknown')}</S.Label>
         <S.FieldWrapper>
            {data.block.type === "data" ? (
               <>
                  {uploadedImage ? (
                     <S.Image
                        src={uploadedImage}
                        alt={data.block.name}
                        onClick={handleClickUpload}
                        style={{ cursor: "pointer" }}
                     />
                  ) : (
                     <S.CustomUploadContainer onClick={handleClickUpload}>
                        <UploadImageIcon width={28} height={28} />
                        <span>Drag files to upload</span>
                     </S.CustomUploadContainer>
                  )}
                  <S.HiddenInput
                     type="file"
                     id="fileUpload"
                     accept="image/jpeg"
                     onChange={handleImageUpload}
                  />
               </>
            ) : (
               blockImage && <S.Image src={blockImage} alt={data.block.name} />
            )}
         </S.FieldWrapper>
         <Handle type="source" position={sourcePosition} />
      </S.Container>
   );
};

export default BlockNodeFeature;
