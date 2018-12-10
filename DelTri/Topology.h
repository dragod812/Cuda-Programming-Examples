

#ifndef TOPOLOGY_INCLUDED
#define TOPOLOGY_INCLUDED

#include "global_datatype.h"
#include "InsidePoints.h"
#include "NNCrust.h"
#include "GetFarthestPoint.h"
#include "DeFace.h"
#include "MarkEdge.h"
#include "ConvToSeg.h"
#include "CreateDelaunay.h"

void Topology(Delaunay& dt_sample, const QuadTree& tree);

#endif //TOPOLOGY_INCLUDED