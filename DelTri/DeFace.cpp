
#include "DeFace.h"

void deFace(Delaunay& dt, const vh& point)
{
	Face_circulator fc = dt.incident_faces(point), done(fc);

	do
	{
		//std::cout<<point->id<<" "<<point->point();
		for (int i = 0; i < 3; ++i)
		{
			Edge e = Edge(fc, i);

			e.first->correct_segments[e.second] = false;
			fh opp_face = e.first->neighbor(e.second);
			int opp_index = opp_face->index(e.first);
			opp_face->correct_segments[opp_index] = false;
		}

	} while (++fc != done);
}