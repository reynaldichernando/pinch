export function euclideanDistance(point1, point2) {
  const deltaX = point2.x - point1.x;
  const deltaY = point2.y - point1.y;
  const deltaZ = point2.z - point1.z;

  const distance = Math.sqrt(deltaX ** 2 + deltaY ** 2 + deltaZ ** 2);
  return distance;
}

export function convertRange(value, oldMin, oldMax, newMin, newMax) {
  const result = (value - oldMin) / (oldMax - oldMin) * (newMax - newMin) + newMin;
  if (result < newMin) return newMin;
  if (result > newMax) return newMax;

  return result;
}