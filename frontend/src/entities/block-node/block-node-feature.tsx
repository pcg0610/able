import { Handle, NodeToolbar, Position } from '@xyflow/react';
import React, { useMemo, useState } from 'react';

import * as S from '@entities/block-node/block-node.style';
import Common from '@shared/styles/common';
import { blockColors } from '@shared/constants/block';
import { capitalizeFirstLetter } from '@shared/utils/formatters.util';
import { BlockItem } from '@/features/canvas/types/block.type';
import { useImageStore } from '@entities/train/model/train.model';

import UploadImageIcon from '@icons/uploadImage.svg?react';
import GraphIcon from '@icons/graph.svg?react';

interface BlockNodeFeatureProps {
  data: {
    block: BlockItem;
    onFieldChange: (fieldName: string, value: string) => void;
    featureMap?: string;
  };
  sourcePosition?: Position;
  targetPosition?: Position;
}

const BlockNodeFeature = ({
  data,
  sourcePosition = Position.Bottom,
  targetPosition = Position.Top,
}: BlockNodeFeatureProps) => {
  const { uploadedImage, setUploadedImage, heatmapImage, classScores, heatMapId } = useImageStore();
  const [isChanged, setIsChanged] = useState(false);

  const blockColor = useMemo(
    () => (data?.block?.type ? blockColors[data.block.type] : Common.colors.gray200),
    [data?.block?.type]
  );

  const blockImage = useMemo(
    () => (data.featureMap ? data.featureMap : null),
    [data.featureMap]
  );

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

  const [isGraphVisible, setIsGraphVisible] = useState(false);
  const toggleGraphVisibility = () => {
    setIsGraphVisible(!isGraphVisible);
  };

  return (
    <S.Container blockColor={blockColor} isConnected isSelected={false}>
      <Handle type="target" position={targetPosition} />
      <S.Label>{capitalizeFirstLetter(data?.block?.name || 'Unknown')}</S.Label>
      <S.FieldWrapper>
        {data.block.type === 'data' ? (
          <>
            {uploadedImage ? (
              <S.Image
                src={uploadedImage}
                alt={data.block.name}
                onClick={handleClickUpload}
                style={{ cursor: 'pointer' }}
              />
            ) : (
              <S.CustomUploadContainer onClick={handleClickUpload}>
                <UploadImageIcon width={28} height={28} />
                <span>Drag files to upload</span>
              </S.CustomUploadContainer>
            )}
            <S.HiddenInput type="file" id="fileUpload" accept="image/jpeg" onChange={handleImageUpload} />
          </>
        ) : data.block.id === heatMapId && heatmapImage ? (
          <div style={{ display: 'flex', alignItems: 'center', position: 'relative' }}>
            <S.Image src={heatmapImage} alt={data.block.name} />
            <S.GraphButton onClick={toggleGraphVisibility}>
              {isGraphVisible ? 'Hidden Graph' : 'Show Graph'}
            </S.GraphButton>
            {isGraphVisible && (
              <NodeToolbar>
                <S.BarContainer>
                  {classScores.map((score, index) => (
                    <S.BarWrapper key={index}>
                      <S.Bar height={score.classScore} color={index === 0 ? '#00274d' : index === 1 ? '#5b8db8' : '#aac4e1'}>
                        <S.BarScore>{score.classScore}</S.BarScore>
                      </S.Bar>
                      <S.BarLabel>{score.className}</S.BarLabel>
                    </S.BarWrapper>
                  ))}
                </S.BarContainer>
              </NodeToolbar>
            )}
          </div>
        ) : (
          uploadedImage && blockImage && <S.Image src={blockImage} alt={data.block.name} />
        )}
      </S.FieldWrapper>
      <Handle type="source" position={sourcePosition} />
    </S.Container>
  );
};

export default BlockNodeFeature;
