#pragma once

#include <vector>
#include <memory>
#include <algorithm>
#include <numeric>
#include <queue>
#include <cmath>

namespace simple_kdtree
{
    namespace trait
    {
        template <typename T, std::size_t D>
        struct access
        {
        };

        template <typename T>
        struct dimension
        {
        };
    } // namespace trait

    template <typename Point, typename DistanceType = double>
    class KDTree
    {
        class KDNode
        {
        public:
            std::size_t index;
            Point point;
            std::unique_ptr<KDNode> left;
            std::unique_ptr<KDNode> right;

            KDNode(std::size_t idx, const Point &pt) : index(idx), point(pt), left(nullptr), right(nullptr) {}
        };

        using PointContainer = std::vector<Point>;
        using IndexContainer = std::vector<std::size_t>;
        using IndexIterator = typename IndexContainer::iterator;
        using NodePtr = std::unique_ptr<KDNode>;
        using HeapData = std::pair<DistanceType, std::size_t>;
        using HeapContainer = std::vector<HeapData>;

    public:
        KDTree(const PointContainer &points)
        {
            IndexContainer indices(points.size());
            std::iota(indices.begin(), indices.end(), 0);
            this->root_ = buildTree<0>(indices.begin(), indices.end(), points);
        }

        void searchKNN(const Point &query, const std::size_t k, std::vector<std::size_t> &out_indices, std::vector<DistanceType> &out_distances) const
        {
            HeapContainer knn_heap;
            knn_heap.reserve(k);
            do_searchKNN<0>(this->root_, query, k, knn_heap);

            out_indices.clear();
            out_distances.clear();
            for (const auto &pair : knn_heap)
            {
                out_indices.push_back(pair.second);
                out_distances.push_back(pair.first);
            }
        }

    private:
        NodePtr root_;

        template <std::size_t axis>
        NodePtr buildTree(const IndexIterator &start, const IndexIterator &end, const PointContainer &points)
        {
            if (start >= end)
            {
                return nullptr;
            }

            constexpr std::size_t dim = simple_kdtree::trait::dimension<Point>::value;
            const auto mid = start + std::distance(start, end) / 2;
            std::nth_element(start, mid, end,
                             [&](const std::size_t lhs, const std::size_t rhs)
                             {
                                 return simple_kdtree::trait::access<Point, axis>::get(points[lhs]) < simple_kdtree::trait::access<Point, axis>::get(points[rhs]);
                             });

            auto node = std::make_unique<KDNode>(*mid, points[*mid]);
            constexpr std::size_t next_axis = (axis + 1) % dim;
            if (start != mid)
            {
                node->left = buildTree<next_axis>(start, mid, points);
            }
            if (mid + 1 != end)
            {
                node->right = buildTree<next_axis>(mid + 1, end, points);
            }

            return node;
        }

        template <std::size_t axis>
        void do_searchKNN(const NodePtr &node, const Point &query, const std::size_t k,
                          HeapContainer &knn_heap) const
        {
            if (!node)
            {
                return;
            }

            constexpr std::size_t dim = simple_kdtree::trait::dimension<Point>::value;
            const auto target = node->point;

            const auto dist = squaredDistance<dim>(target, query);

            if (knn_heap.size() < k)
            {
                knn_heap.emplace_back(dist, node->index);
                std::push_heap(knn_heap.begin(), knn_heap.end(), std::greater<>());
            }
            else if (dist < knn_heap.front().first)
            {
                std::pop_heap(knn_heap.begin(), knn_heap.end(), std::greater<>());
                knn_heap.back() = {dist, node->index};
                std::push_heap(knn_heap.begin(), knn_heap.end(), std::greater<>());
            }

            const auto diff = getCoordinate<axis>(query) - getCoordinate<axis>(target);
            constexpr std::size_t next_axis = (axis + 1) % dim;

            if (diff < 0)
            {
                do_searchKNN<next_axis>(node->left, query, k, knn_heap);
                if (knn_heap.size() >= k && std::abs(diff) >= std::sqrt(knn_heap.front().first))
                {
                    return;
                }
                do_searchKNN<next_axis>(node->right, query, k, knn_heap);
            }
            else
            {
                do_searchKNN<next_axis>(node->right, query, k, knn_heap);
                if (knn_heap.size() >= k && std::abs(diff) >= std::sqrt(knn_heap.front().first))
                {
                    return;
                }
                do_searchKNN<next_axis>(node->left, query, k, knn_heap);
            }
        }

        template <std::size_t I>
        auto getCoordinate(const Point &point) const
        {
            return simple_kdtree::trait::access<Point, I>::get(point);
        }

        template <std::size_t dim>
        DistanceType squaredDistance(const Point &a, const Point &b) const
        {
            if constexpr (dim == 1)
            {
                const auto dx = getCoordinate<0>(a) - getCoordinate<0>(b);
                return dx * dx;
            }
            else if constexpr (dim == 2)
            {
                const auto dx = getCoordinate<0>(a) - getCoordinate<0>(b);
                const auto dy = getCoordinate<1>(a) - getCoordinate<1>(b);
                return dx * dx + dy * dy;
            }
            else if constexpr (dim == 3)
            {
                const auto dx = getCoordinate<0>(a) - getCoordinate<0>(b);
                const auto dy = getCoordinate<1>(a) - getCoordinate<1>(b);
                const auto dz = getCoordinate<2>(a) - getCoordinate<2>(b);
                return dx * dx + dy * dy + dz * dz;
            }
            else
            {
                const auto dx = getCoordinate<0>(a) - getCoordinate<0>(b);
                const auto dy = getCoordinate<1>(a) - getCoordinate<1>(b);
                const auto dz = getCoordinate<2>(a) - getCoordinate<2>(b);
                const auto dw = getCoordinate<3>(a) - getCoordinate<3>(b);
                return dx * dx + dy * dy + dz * dz + dw * dw;
            }
        }
    };
} // namespace simple_kdtree
