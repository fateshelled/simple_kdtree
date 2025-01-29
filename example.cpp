#include <iostream>
#include <vector>
#include "simple_kdtree.h"

struct point2d
{
    double x, y;
};

struct point3d
{
    double x, y, z;
};

template <std::size_t D>
struct simple_kdtree::trait::access<point2d, D>
{
    static auto get(const point2d &p) -> float
    {
        if constexpr (D == 0)
        {
            return p.x;
        }
        else
        {
            return p.y;
        }
    }
};

template <>
struct simple_kdtree::trait::dimension<point2d>
{
    static constexpr std::size_t value = 2;
};

template <std::size_t D>
struct simple_kdtree::trait::access<point3d, D>
{
    static auto get(const point3d &p) -> float
    {
        if constexpr (D == 0)
        {
            return p.x;
        }
        else if constexpr (D == 1)
        {
            return p.y;
        }
        else
        {
            return p.z;
        }
    }
};

template <>
struct simple_kdtree::trait::dimension<point3d>
{
    static constexpr std::size_t value = 3;
};

int main()
{
    // 2D Points
    {
        const std::vector<point2d> points = {{1, 2}, {3, 4}, {5, 6}};

        // Create a KDTree 2D
        const simple_kdtree::KDTree<point2d> tree(points);

        // Search
        const point2d query = {4, 3};
        std::vector<size_t> indices;
        std::vector<double> sq_distances;
        tree.searchKNN(query, 2, indices, sq_distances);

        // Print result.
        std::cout << "Query Point: (" << query.x << ", " << query.y << ")" << std::endl;
        std::cout << "Nearest Point: (" << points[indices[0]].x << ", " << points[indices[0]].y << "), distance " << std::sqrt(sq_distances[0]) << std::endl;
        std::cout << "Second Point: (" << points[indices[1]].x << ", " << points[indices[1]].y << "), distance " << std::sqrt(sq_distances[1]) << std::endl;
        std::cout << std::endl;
    }

    // 3D Points
    {
        const std::vector<point3d> points = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

        // Create a KDTree 3D
        const simple_kdtree::KDTree<point3d> tree(points);

        // Search
        const point3d query = {6, 7, 8};
        std::vector<size_t> indices;
        std::vector<double> sq_distances;
        tree.searchKNN(query, 2, indices, sq_distances);

        // Print result.
        std::cout << "Query Point: (" << query.x << ", " << query.y << ", " << query.z << ")" << std::endl;
        std::cout << "Nearest Point: (" << points[indices[0]].x << ", " << points[indices[0]].y << ", " << points[indices[0]].z << "), distance " << std::sqrt(sq_distances[0]) << std::endl;
        std::cout << "Second Point: (" << points[indices[1]].x << ", " << points[indices[1]].y << ", " << points[indices[1]].z << "), distance " << std::sqrt(sq_distances[1]) << std::endl;
        std::cout << std::endl;
    }

    return 0;
}
