import { ReactNode, useEffect } from 'react';
import ReactDOM from 'react-dom';

interface ModalPortalProps {
  children: ReactNode;
}

const ModalPortal = ({ children }: ModalPortalProps) => {
  // 포털의 루트가 되는 `#modal-root`를 가져옴
  const modalRoot = document.getElementById('modal-root');

  // 모달을 렌더링할 임시 DOM 노드를 생성
  const el = document.createElement('div');

  useEffect(() => {
    // 모달이 열릴 때 `#modal-root`에 임시 DOM 노드를 추가
    modalRoot?.appendChild(el);

    // 모달이 닫힐 때 임시 노드를 제거
    return () => {
      modalRoot?.removeChild(el);
    };
  }, [el, modalRoot]);

  // children을 임시 노드에 렌더링
  return ReactDOM.createPortal(children, el);
};

export default ModalPortal;
