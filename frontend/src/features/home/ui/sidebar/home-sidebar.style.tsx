import styled from '@emotion/styled';

export const SidebarContainer = styled.div`
  width: 250px; // 사이드바의 고정 너비
  height: 100vh; // 화면 전체 높이
  background-color: #f8f9fc; // 배경색 설정
  padding: 20px; // 전체적인 여백
  border-right: 1px solid #ddd; // 오른쪽 경계선
`;

export const Title = styled.h2`
  font-size: 16px;
  font-weight: bold;
  margin-bottom: 4px;
`;

export const Subtitle = styled.p`
  font-size: 12px;
  color: #6c757d;
  margin-bottom: 16px;
`;

export const Button = styled.button`
  display: block;
  width: 100%;
  padding: 10px;
  background-color: #e7f3ff;
  border: none;
  border-radius: 8px;
  color: #007bff;
  font-weight: bold;
  text-align: left;
  margin-bottom: 16px;
  cursor: pointer;
`;

export const FolderSection = styled.div`
  margin-bottom: 16px;
`;

export const Folder = styled.div`
  display: flex;
  align-items: center;
  font-size: 14px;
  color: #343a40;
  margin-bottom: 8px;
  cursor: pointer;
`;

export const Icon = styled.span`
  margin-right: 8px;
`;

export const Footer = styled.div`
  bottom: 20px;
  left: 20px;
  right: 20px;
  display: flex;
  align-items: center;
  background-color: #e7f3ff;
  border-radius: 8px;
  padding: 10px;
`;

export const FooterIcon = styled.span`
  margin-right: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
`;

export const FooterText = styled.div`
  font-size: 14px;
  font-weight: bold;
  color: #343a40;
`;

export const FooterStatus = styled.div`
  font-size: 12px;
  color: #6c757d;
`;