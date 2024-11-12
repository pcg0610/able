const ConfusionMatrix = ({ confusionMatrix }: { confusionMatrix?: string }) => {
  if (!confusionMatrix) {
    return null;
  }

  return (
    <img
      src={confusionMatrix}
      alt="Confusion Matrix"
      style={{
        width: '100%', maxHeight: '100%', objectFit: 'contain', transform: 'scale(0.8)', // 이미지 스케일을 조정하여 크기 축소
        transformOrigin: 'center',
      }} // 부모의 높이에 맞춰서 이미지 표시
    />
  );
};

export default ConfusionMatrix;