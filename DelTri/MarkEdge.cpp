
#include "MarkEdge.h"


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

void check_duplicate(std::vector<Point_2> &input)
{
	//std::cout.precision(17);
	//std::vector<Point_2> th_pts;
	for (int j = 0; j<input.size(); ++j)
	{
		for (int k = j; k<input.size(); ++k)
		{
			if (j == k){ continue; }
			else
			{
				if (input.at(j) == input.at(k))
				{
					//ThreshPoints.erase(ThreshPoints.begin(),ThreshPoints.begin()+k);	--k;
					input.erase(input.begin() + k);	--k;
				}

			}
		}
	}
}


void markEdge(Delaunay& dt, const QuadTree& tree, const vh& point)
{
	//std::ofstream HomoEdges;
	//HomoEdges.open("HomomorphicalEdges.txt",std::ofstream::out | std::ofstream::trunc);
	//HomoEdges.precision(17);
	Face_circulator fc = dt.incident_faces(point), done(fc);
	do
	{
		//std::cout<<point->id<<" "<<point->point();
		for (int i = 0; i < 3; ++i)
		{
			Edge e = Edge(fc, i);
			if (!dt.is_infinite(e))
			{
				CGAL::Object o = dt.dual(e);

				const Segment_2* s = CGAL::object_cast<Segment_2>(&o);
				const Ray_2* r = CGAL::object_cast<Ray_2>(&o);

				int num_of_intersections = 0;
				Segment_2* temp = new Segment_2;
				ThreshPoints.clear(); NewThreshPoints.clear(); Neighbors.clear(); Neighbor_Segments.clear();

				if (r)
				{
					if (tree.rootNode->rectangle.has_on_bounded_side((*r).source())){
						*temp = convToSeg(tree.rootNode->rectangle, *r);
					}
				}
				if (s)
				{
					*temp = *s;
				}
				ThreshPoints = insidePoints(tree, *temp, OT);
				check_duplicate(ThreshPoints);

				Delaunay dt_thresh;
				create_Delaunay(dt_thresh, ThreshPoints);

				NewThreshPoints = insidePoints(tree, *temp, IT);
				check_duplicate(NewThreshPoints);

				//std::cout<<"Thresh Points "<<ThreshPoints.size()<<"		NewThreshPoints		"<<NewThreshPoints.size()<<std::endl;
				if (ThreshPoints.size() > 2 && NewThreshPoints.size() > 1)
				{
					NNCrust(dt_thresh, NewThreshPoints, Neighbors, Neighbor_Segments, *temp, IT);
					num_of_intersections = check_Intersection(*temp, Neighbor_Segments);
					//n = n + num_of_intersections;
					//total_inEdgeMarking = total_inEdgeMarking + num_of_intersections;			
					//std::cout<<"total_inEdgeMarking = "<<total_inEdgeMarking<<std::endl;
					if (num_of_intersections == 1)
					{
						e.first->correct_segments[e.second] = true;
						fh opp_face = e.first->neighbor(e.second);
						int opp_index = opp_face->index(e.first);
						opp_face->correct_segments[opp_index] = true;
					}
				}
			}
		}
	} while (++fc != done);
	//}
	/*for(Edge_iterator eg = dt.finite_edges_begin(); eg != dt.finite_edges_end(); ++eg)
	{
	if((eg->first->correct_segments[eg->second] == true))

	{
	HomoEdges<<eg->first->vertex((eg->second+1)%3)->point()<<" "<<eg->first->vertex((eg->second+2)%3)->point()<<std::endl;
	//std::cout<<eg->first->vertex((eg->second+1)%3)->point()<<" "<<eg->first->vertex((eg->second+2)%3)->point()<<std::endl;
	}
	}*/
}