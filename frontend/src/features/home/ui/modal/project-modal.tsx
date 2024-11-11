import { useState, useEffect } from 'react';
import toast from 'react-hot-toast';

import { useProjectStore } from '@entities/project/model/project.model';
import { useCreateProject, useUpdateProject, useDeleteProject } from '@features/home/api/use-home.mutation';

import Modal from '@shared/ui/modal/modal';
import Input from '@shared/ui/input/input';
import DropDown from '@shared/ui/dropdown/dropdown';

interface ProjectModalProps {
  onClose: () => void;
  isClosing: boolean;
  onAnimationEnd: () => void;
  type: 'create' | 'modify';
}

const ProjectModal = ({ onClose, isClosing, onAnimationEnd, type }: ProjectModalProps) => {
  const isReadOnly = type === 'modify';
  const { currentProject } = useProjectStore();
  const [selectedOption, setSelectedOption] = useState<string>(null);
  const [projectTitle, setProjectTitle] = useState(currentProject?.title || '');
  const [projectDescription, setProjectDescription] = useState(currentProject?.description || '');
  const [projectCudaVersion, setProjectCudaVersion] = useState(currentProject?.cudaVersion || '');
  const [pythonKernelPath, setPythonKernelPath] = useState(currentProject?.pythonKernelPath || '');
  const { mutate: createProject } = useCreateProject();
  const { mutate: updateProject } = useUpdateProject();
  const { mutate: deleteProject } = useDeleteProject();

  const options = [
    { value: 'option0', label: 'Option 0' },
    { value: 'option1', label: 'Option 1' },
    { value: 'option2', label: 'Option 2' },
    { value: 'option3', label: 'Option 3' },
    { value: 'option4', label: 'Option 4' },
    { value: 'option5', label: 'Option 5' },
  ];

  useEffect(() => {
    if (isReadOnly && currentProject) {
      setProjectTitle(currentProject.title || '');
      setProjectDescription(currentProject.description || '');
      setProjectCudaVersion(currentProject.cudaVersion || '');
      setPythonKernelPath(currentProject.pythonKernelPath || '');
    } else {
      setProjectTitle('');
      setProjectDescription('');
      setProjectCudaVersion('');
      setPythonKernelPath('');
    }
  }, [isReadOnly, currentProject]);

  const handleSelect = (option: string) => {
    setSelectedOption(option);
  };

  const handleCreateProject = () => {
    if (!isReadOnly) {
      createProject(
        {
          title: projectTitle,
          description: projectDescription,
          cudaVersion: selectedOption,
          pythonKernelPath: pythonKernelPath,
        },
        {
          onSuccess: (data) => {
            if (data) {
              toast.success("프로젝트가 생성되었습니다.");
              onClose();
            }
          },
        }
      );
    } else {
      updateProject(
        {
          title: projectTitle,
          description: projectDescription,
          prevTitle: currentProject?.title,
          prevDescription: currentProject?.description,
        },
        {
          onSuccess: (data) => {
            if (data) {
              toast.success("프로젝트 정보가 수정되었습니다.");
              onClose();
            }
          },
        }
      );
    }
  };

  const handleDeleteProject = () => {
    deleteProject(
      { title: projectTitle },
      {
        onSuccess: (data) => {
          if (data) {
            toast.success("프로젝트가 삭제되었습니다.");
            onClose();
          } else {
            toast.error("오류가 발생했습니다.");
            onClose();
          }
        },
      }
    );
  }

  return (
    <Modal
      onClose={onClose}
      onDelete={handleDeleteProject}
      onConfirm={handleCreateProject}
      isClosing={isClosing}
      onAnimationEnd={onAnimationEnd}
      title="프로젝트 정보를 입력하세요"
      confirmText={isReadOnly ? '수정' : '생성'}
      cancelText={'삭제'}
      isDelete={true}
    >
      <Input
        label="프로젝트 이름"
        value={projectTitle}
        placeholder={isReadOnly ? '' : '2-50자 이내로 입력해주세요.'}
        onChange={(e) => setProjectTitle(e.target.value)}
      />
      <Input
        label="프로젝트 설명 (선택)"
        value={projectDescription}
        placeholder={isReadOnly ? '' : '2-50자 이내로 입력해주세요.'}
        onChange={(e) => setProjectDescription(e.target.value)}
      />
      <Input
        label="파이썬 커널 경로"
        defaultValue={isReadOnly ? currentProject?.pythonKernelPath : ''}
        readOnly={isReadOnly}
        className={isReadOnly ? 'readonly' : ''}
        placeholder=".exe"
        onChange={(e) => setPythonKernelPath(e.target.value)}
      />
      {isReadOnly ? <Input
        label="쿠다 버전"
        defaultValue={currentProject?.cudaVersion}
        readOnly={isReadOnly}
        className={'readonly'}
      /> :
        <DropDown
          label="쿠다 버전"
          options={options}
          onSelect={handleSelect}
        />
      }
    </Modal>
  );
};

export default ProjectModal;
