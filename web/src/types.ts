export type Point = [number, number];

export interface Person {
  id: number;
  bbox: [number, number, number, number];
  head_center: Point;
  body_center: Point;
  confidence: number;
}

export interface Density {
  grid_size: [number, number];
  cells: number[][];
  max_cell: [number, number];
}

export interface FramePayload {
  frame_id: number;
  timestamp: number;
  persons: Person[];
  density: Density;
  fps: number;
  frame_size: [number, number];
}

export interface StatsPayload {
  total_persons: number;
  fps: number;
  densest_cell?: [number, number];
  error?: string | null;
}
