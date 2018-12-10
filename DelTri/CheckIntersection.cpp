
//#include "CheckIntersection.h"
#include "global_datatype.h"

//Function to check number of intersections
//template <typename Type1,typename Type2>
int check_Intersection(Segment_2 rays, std::vector<Segment_2> seg)
{
	//std::cout.precision(17);
	int count = 0;
	bool on_theRay = false;
	for (int j = 0; j<seg.size(); j++)
	{
		if ((seg.at(j).source().x()<DBL_MAX) || (seg.at(j).source().y()<DBL_MAX) || (seg.at(j).target().x()<DBL_MAX) || (seg.at(j).target().y()<DBL_MAX))
		{
			if (seg.at(j).source() != seg.at(j).target())
			{
				if (CGAL::do_intersect(rays, seg.at(j)))
				{
					++count;
					//points.push_back(CGAL::intersection(rays,seg.at(j));
				}
				if ((rays.has_on(seg.at(j).source())) || (rays.has_on(seg.at(j).target()))){ on_theRay = true; }

			}
		}
	}
	if (on_theRay){ --count; }
	return count;
}
