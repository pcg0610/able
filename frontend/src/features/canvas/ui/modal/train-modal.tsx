import { useState } from 'react';

import Modal from '@shared/ui/modal/modal';
import Input from '@shared/ui/input/input';
import DropDown from '@shared/ui/dropdown/dropdown';

interface TrainModalProps {
  onClose: () => void;
}

interface TrainConfig {
  epoch: number | null;
  batchSize: number | null;
  device: { index: number | null; name: string };
}

const TrainModal = ({ onClose }: TrainModalProps) => {
  const [trainingConfig, setTrainingConfig] = useState<TrainConfig>({
    epoch: null,
    batchSize: null,
    device: { index: null, name: '' },
  });

  const handleInputChange = (field: keyof TrainConfig, value: number) => {
    setTrainingConfig((prev) => ({
      ...prev,
      [field]: value,
    }));
  };

  return (
    <Modal onClose={onClose} title="학습 정보를 입력하세요" confirmText="학습하기">
      <Input
        label="학습 횟수 (epoch)"
        value={trainingConfig.epoch || ''}
        onChange={(e) => handleInputChange('epoch', Number(e.target.value))}
      />
      <Input
        label="배치 크기 (batch size)"
        value={trainingConfig.batchSize || ''}
        onChange={(e) => handleInputChange('batchSize', Number(e.target.value))}
      />
      <DropDown
        label="학습 장치 선택"
        options={[
          { label: 'CPU', value: 'CPU' },
          { label: 'GPU', value: 'GPU' },
        ]}
        onSelect={(option) =>
          setTrainingConfig((prev) => ({
            ...prev,
            device: { index: option.value === 'CPU' ? 0 : 1, name: option.value }, // 수정된 부분
          }))
        }
      />
    </Modal>
  );
};

export default TrainModal;
