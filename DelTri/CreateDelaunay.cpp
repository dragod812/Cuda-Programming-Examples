//#include "global_datatype.h"
#include "CreateDelaunay.h"

void create_Delaunay(Delaunay& dt, std::vector<Point_2> &input)
{
	dt.insert(input.begin(), input.end());
}
