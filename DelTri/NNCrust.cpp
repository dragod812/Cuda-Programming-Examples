
#include "NNCrust.h"

//Function to get nearest and next_nearest neighbors using NNCrust and store it in a vector in the form {point,nearest,next_nearest}
//template <typename Type>
void NNCrust(Delaunay& dt, std::vector<Point_2>& sample, std::vector<vector <Point_2> >& Neighbors, std::vector<Segment_2>& seg, Segment_2& ray, const double& threshold)
{
	//std::cout.precision(17);
	Finite_vertices_iterator_2d vit;
	std::vector< vector <Point_2> > TestNeigh;

	for (vit = dt.finite_vertices_begin(); vit != dt.finite_vertices_end(); vit++)
	{
		bool notThresh_point = true;
		if (CGAL::squared_distance(ray, vit->point())<threshold)
		{
			notThresh_point = false;
		}
		else{ notThresh_point = true; }

		if (notThresh_point == false)
		{
			int count = 0;
			//store pair of vertex 
			//for each vertex find the Delaunay neighbors.
			std::vector<vh> neighbors;
			std::vector<Point_2> TestP;
			TestP.push_back(vit->point());
			//get a face handle incident on vit
			fh f1 = vit->face();

			//find the index of vit in f1
			int index = f1->index(vit);
			int index1 = (index + 1) % 3;
			int index2 = (index + 2) % 3;

			Vertex_circulator vcirculator = dt.incident_vertices(vit), done(vcirculator);
			do
			{
				if (!dt.is_infinite(vcirculator)){ neighbors.push_back(vcirculator); }
			} while (++vcirculator != done);

			for (int i = 0; i<neighbors.size(); i++)
			{
				TestP.push_back(neighbors[i]->point());
			}
			TestNeigh.push_back(TestP);
		}
	}


	for (int l = 0; l<TestNeigh.size(); ++l)
	{
		int nearest = 0; int next_nearest = 0;
		double d_min = DBL_MAX, d_min1 = DBL_MAX;
		for (int j = 1; j<TestNeigh[l].size(); ++j)
		{
			double d = (squared_distance(TestNeigh[l].at(0), TestNeigh[l].at(j)));
			if (d<d_min)
			{
				d_min = d;
				nearest = j;
			}
		}
		for (int k = 1; k<TestNeigh[l].size(); ++k)
		{
			Point p1 = TestNeigh[l].at(0);
			Point p2 = TestNeigh[l].at(nearest);
			Point p3 = TestNeigh[l].at(k);
			double dist = (squared_distance(p1, p3));

			if (((p2 != p3) && (p1 != p3)) && (CGAL::angle(p2, p1, p3) != CGAL::ACUTE))
			{

				if (dist < d_min1)
				{
					d_min1 = dist;
					next_nearest = k;
				}
				//std::cout<<TestNeigh[l].at(0)<<" "<<TestNeigh[l].at(k)<<" "<<next_nearest<<" "<<k<<" "<<dist<<" "<<d_min1<<" Distance "<<std::endl;
			}
		}
		std::vector<Point_2> TempNeighbors;
		TempNeighbors.push_back(TestNeigh[l].at(0));
		TempNeighbors.push_back(TestNeigh[l].at(nearest));

		TempNeighbors.push_back(TestNeigh[l].at(next_nearest));
		Neighbors.push_back(TempNeighbors);
	}
	for (int p = 0; p<Neighbors.size(); p++)
	{
		Segment_2 segment1(Neighbors[p].at(0), Neighbors[p].at(1));
		Segment_2 segment2(Neighbors[p].at(0), Neighbors[p].at(2));
		seg.push_back(segment1);
		seg.push_back(segment2);

		for (int q = 0; q<seg.size() - 1; q++)
		{
			for (int r = q; r<seg.size(); r++)
			{
				if (((seg.at(q).source() == seg.at(r).target()) && (seg.at(q).target() == seg.at(r).source())) || ((seg.at(r).target().x() - seg.at(r).source().x()<0.05) && (seg.at(r).target().y() - seg.at(r).source().y()<0.05)))
				{
					seg.erase(seg.begin() + r);
				}
			}
		}
	}
}