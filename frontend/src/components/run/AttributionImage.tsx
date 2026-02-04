import React, { useEffect, useRef } from "react";

type AttributionImageProps = {
  imageUrl?: string;
  heatmap?: number[][] | number[];
};

const AttributionImage: React.FC<AttributionImageProps> = ({ imageUrl, heatmap }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const img = imgRef.current;
    if (!canvas || !img || !heatmap) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const draw = () => {
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const data = Array.isArray(heatmap[0]) ? (heatmap as number[][]) : [];
      if (data.length === 0) return;
      const h = data.length;
      const w = data[0].length;
      const cellW = canvas.width / w;
      const cellH = canvas.height / h;
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          const value = data[y][x] ?? 0;
          ctx.fillStyle = `rgba(255, 50, 50, ${value})`;
          ctx.fillRect(x * cellW, y * cellH, cellW, cellH);
        }
      }
    };

    if (img.complete) {
      draw();
    } else {
      img.onload = draw;
    }
  }, [heatmap, imageUrl]);

  return (
    <div className="image-overlay">
      {imageUrl ? <img ref={imgRef} src={imageUrl} alt="input" /> : <div>No image</div>}
      <canvas ref={canvasRef} className="heatmap-canvas" />
    </div>
  );
};

export default AttributionImage;
