#pragma once

#include <limits>
#include <vector>

struct Point {
  double x, y;
};

struct Triangle {
  size_t p1, p2, p3;

  bool operator==(const Triangle& other) const {
    return (p1 == other.p1 && p2 == other.p2 && p3 == other.p3) ||
           (p1 == other.p2 && p2 == other.p3 && p3 == other.p1) ||
           (p1 == other.p3 && p2 == other.p1 && p3 == other.p2);
  }
};

bool isInCircle(const Point& p, const Point& p1, const Point& p2,
                const Point& p3) {
  double A = p2.x - p1.x;
  double B = p2.y - p1.y;
  double C = p3.x - p1.x;
  double D = p3.y - p1.y;

  double det = A * D - B * C;
  double s = A * A + B * B;
  double t = C * C + D * D;

  double cx = (D * s - B * t) / (2 * det);
  double cy = (A * t - C * s) / (2 * det);
  double r = std::sqrt(cx * cx + cy * cy);

  double dx = p.x - cx;
  double dy = p.y - cy;

  return dx * dx + dy * dy <= r * r;
}

bool isPointInTriangle(const Point& p, const Point& p1, const Point& p2,
                       const Point& p3) {
  double ABC = (p1.x - p.x) * (p2.y - p.y) - (p2.x - p.x) * (p1.y - p.y);
  double BCA = (p2.x - p.x) * (p3.y - p.y) - (p3.x - p.x) * (p2.y - p.y);
  double CAB = (p3.x - p.x) * (p1.y - p.y) - (p1.x - p.x) * (p3.y - p.y);

  return (ABC >= 0 && BCA >= 0 && CAB >= 0) ||
         (ABC <= 0 && BCA <= 0 && CAB <= 0);
}

void addSuperTriangle(std::vector<Point>& points,
                      std::vector<Triangle>& triangles) {
  double maxX = std::numeric_limits<double>::min();
  double maxY = std::numeric_limits<double>::min();
  double minX = std::numeric_limits<double>::max();
  double minY = std::numeric_limits<double>::max();

  for (const auto& p : points) {
    maxX = std::max(maxX, p.x);
    maxY = std::max(maxY, p.y);
    minX = std::min(minX, p.x);
    minY = std::min(minY, p.y);
  }

  double dx = maxX - minX;
  double dy = maxY - minY;
  double delta = std::max(dx, dy) * 20.0;

  Point p1(minX - delta, minY - delta);
  Point p2(minX - delta, maxY + 2 * delta);
  Point p3(maxX + 2 * delta, minY - delta);

  points.push_back(p1);
  points.push_back(p2);
  points.push_back(p3);

  triangles.push_back(
      Triangle(points.size() - 3, points.size() - 2, points.size() - 1));
}

std::vector<Triangle> delaunayTriangulation(std::vector<Point> points) {
  std::vector<Triangle> triangles;
  if (points.size() < 3) return triangles;

  addSuperTriangle(points, triangles);

  for (size_t i = 0; i < points.size(); ++i) {
    std::vector<Triangle> badTriangles;
    for (const auto& t : triangles) {
      const Point& p1 = points[t.p1];
      const Point& p2 = points[t.p2];
      const Point& p3 = points[t.p3];
      if (isInCircle(points[i], p1, p2, p3)) {
        badTriangles.push_back(t);
      }
    }

    std::vector<std::pair<int, int>> polygonEdges;
    for (const auto& t : badTriangles) {
      polygonEdges.emplace_back(t.p1, t.p2);
      polygonEdges.emplace_back(t.p2, t.p3);
      polygonEdges.emplace_back(t.p3, t.p1);
    }

    for (const auto& t : badTriangles) {
      auto it = std::find(triangles.begin(), triangles.end(), t);
      if (it != triangles.end()) {
        triangles.erase(it);
      }
    }

    for (const auto& edge : polygonEdges) {
      if (std::count(polygonEdges.begin(), polygonEdges.end(), edge) == 1) {
        triangles.push_back(Triangle(edge.first, edge.second, i));
      }
    }
  }

  triangles.erase(std::remove_if(triangles.begin(), triangles.end(),
                                 [&](const Triangle& t) {
                                   return t.p1 >= points.size() - 3 ||
                                          t.p2 >= points.size() - 3 ||
                                          t.p3 >= points.size() - 3;
                                 }),
                  triangles.end());

  return triangles;
}
