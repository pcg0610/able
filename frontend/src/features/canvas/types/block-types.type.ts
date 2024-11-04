import { BLOCK_MENU } from '@features/canvas/costants/block-types.constant';

export type MenuName = (typeof BLOCK_MENU)[number]['name'];

interface BlockField {
  name: string;
  value: string;
  isRequired: boolean;
}

export interface Block {
  name: string;
  type: string;
  args: BlockField[];
}

export interface BlocksResponse {
  status_code: number;
  timeStamp: string;
  trackingId: string;
  data: { blocks: Block[] };
}
