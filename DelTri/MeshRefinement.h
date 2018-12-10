

#ifndef MESHREFINEMENT_INCLUDED
#define MESHREFINEMENT_INCLUDED

#include "global_datatype.h"
#include "Topology.h"
#include "Geometry.h"
#include "Skinny.h"
#include "CreateDelaunay.h"

void MeshRefinement(Delaunay& dt_sample, const QuadTree& tree);

#endif //MESHREFINEMENT_INCLUDED