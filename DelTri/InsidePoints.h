

#ifndef INSIDEPOINTS_INCLUDED
#define INSIDEPOINTS_INCLUDED
#include "global_datatype.h"


bool ClipTest(double p, double q, double *u1, double *u2);
std::vector<Segment_2> bBox(const Segment_2& edge, const double& thresh);

bool intersect(const Segment_2& ray, const Iso_rectangle_2& box);

template <typename Type1, typename Type2, typename Type3>
void thresh_Points(const Type1& Object1, const Type2& Object2, const double& threshold, Type3& Object3);


std::vector<Point_2> insidePoints(const QuadTree& tree, const Segment_2& edge, const double& thresh);


#endif //INSIDEPOINTS_INCLUDED