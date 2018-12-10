
#ifndef NNCRUST_INCLUDED
#define NNCRUST_INCLUDED
#include "global_datatype.h"

void NNCrust(Delaunay& dt, std::vector<Point_2>& sample, std::vector<vector <Point_2> >& Neighbors, std::vector<Segment_2>& seg, Segment_2& ray, const double& threshold);

#endif //NNCRUST_INCLUDED