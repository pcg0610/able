import { BLOCK_MENU } from '@features/canvas/costants/block-types.constant';

export type MenuName = (typeof BLOCK_MENU)[number]['name'];

// 블록 노드의 정보를 전달하는 Item
export interface BlockItem {
  type: MenuName;
  name: string;
  fields: BlockField[];
}

export interface BlockField {
  name: string;
  value: string;
  isRequired: boolean;
}

export interface Block {
  name: string;
  type: MenuName;
  args: BlockField[];
}

export interface BlocksResponse {
  status_code: number;
  timeStamp: string;
  trackingId: string;
  data: { blocks: Block[] };
}
