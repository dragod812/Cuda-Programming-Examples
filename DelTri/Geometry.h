

#ifndef GEOMETRY_INCLUDED
#define GEOMETRY_INCLUDED

#include "global_datatype.h"
//#include "Topology.h"
#include "NNCrust.h"
#include "InsidePoints.h"
#include "GetFarthestPoint.h"
//#include "CheckDuplicate.h"
//#include "CheckIntersection.h"
#include "DeFace.h"
#include "MarkEdge.h"
#include "ConvToSeg.h"
#include "CreateDelaunay.h"

void Geometry(Delaunay& dt_sample, const QuadTree& tree);

#endif //GEOMETRY_INCLUDED