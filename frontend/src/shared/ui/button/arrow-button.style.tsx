import styled from '@emotion/styled';

export const StyledButton = styled.div<{
  direction: 'up' | 'down' | 'left' | 'right';
  size: 'md' | 'sm';
}>`
  width: ${({ size }) => (size === 'md' ? '1.5rem' : '1rem')};
  height: ${({ size }) => (size === 'md' ? '1.5rem' : '1rem')};
  display: flex;
  align-items: center;
  justify-content: center;

  transform: ${({ direction }) => {
    switch (direction) {
      case 'up':
        return 'rotate(90deg)';
      case 'left':
        return 'rotate(0deg)';
      case 'down':
        return 'rotate(-90deg)';
      case 'right':
        return 'rotate(180deg)';
      default:
        return 'rotate(0deg)';
    }
  }};
`;
