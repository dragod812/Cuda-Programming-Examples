

#ifndef MARKEDGE_INCLUDED
#define MARKEDGE_INCLUDED

#include "global_datatype.h"
#include "InsidePoints.h"
//#include "CheckDuplicate.h"
#include "NNCrust.h"
//#include "GetFarthestPoint.h"
//#include "CheckIntersection.h"
//#include "DeFace.h"
//#include "MarkEdge.h"
#include "ConvToSeg.h"
#include "CreateDelaunay.h"

int check_Intersection(Segment_2 rays, std::vector<Segment_2> seg);
void check_duplicate(std::vector<Point_2> &input);
void markEdge(Delaunay& dt, const QuadTree& tree, const vh& point);

#endif //MARKEDGE_INCLUDED