
#include "Geometry.h"
#include "Topology.h"
/*std::vector<Point_2> ThreshPoints, NewThreshPoints;
std::vector<vector <Point_2> > Neighbors;
std::vector<Segment_2> Neighbor_Segments;
std::multimap<Edge, Point_2> listing;
*/


void Geometry(Delaunay& dt_sample, const QuadTree& tree)
{
	bool iterate = true; int no = 0;
	int check;
	
	//std::ofstream outputRays;
	//outputRays.open("OutputRays.txt", std::ofstream::out | std::ofstream::trunc);
	
	Edge_iterator eit = dt_sample.finite_edges_begin();
	/*std::cout << ".....................................Restricted Edge after Topology............................." << std::endl;
	for (; eit != dt_sample.finite_edges_end(); ++eit)
	{  //5

		//std::cout << ".....................................Edge............................." << std::endl;
		if (eit->first->correct_segments[eit->second] == true)
		{  //6


			std::cout << eit->first->vertex((eit->second + 1) % 3)->point() << " " << eit->first->vertex((eit->second + 2) % 3)->point() << std::endl;


		}
	}*/ //For Paper

	std::cout << "-----------------Geometry---------------------" << std::endl;
	while (iterate)
	{
		listing.clear();
		Point_2 Far_Point;
		
		for (eit = dt_sample.finite_edges_begin(); eit != dt_sample.finite_edges_end(); ++eit)
		{
			if (eit->first->correct_segments[eit->second] == true){
				
				iterate = false;
				for (int i = 1; i < 3; ++i)
				{//7	
					//std::cout << "2" << std::endl;
					int count = 0;
					std::vector<Edge_circulator> edge_cir;
					vh vh_center = eit->first->vertex((eit->second + i) % 3);
					vh vh_adj, vh_adj1;

					Edge_circulator circulator = dt_sample.incident_edges(vh_center), done(circulator);
					//std::cout << "vh_center	" << vh_center->point() << std::endl;
					//if (no == 0) std::cout<<"vh_center	"<<vh_center->point()<<std::endl;   // For Paper
					do
					{ //8
						//std::cout << "3" << std::endl;
						if (circulator->first->correct_segments[circulator->second] == true)
						{  //9
							if (vh_center == circulator->first->vertex((circulator->second + 1) % 3))
							{
								if (count < 1)
								{
									vh_adj = circulator->first->vertex((circulator->second + 2) % 3);
									/*if (no == 0)  std::cout << "vh_adj	" << vh_adj->point() << std::endl;*/
								}
								if (count == 1)
								{
									vh_adj1 = circulator->first->vertex((circulator->second + 2) % 3);
									/*if (no == 0) std::cout << "vh_adj1	" << vh_adj1->point() << std::endl; */
								}
							}
							if (vh_center == circulator->first->vertex((circulator->second + 2) % 3))
							{
								if (count < 1)
								{
									vh_adj = circulator->first->vertex((circulator->second + 1) % 3);
									/*if (no == 0) std::cout << "vh_adj	" << vh_adj->point() << std::endl;*/
								}
								if (count == 1)
								{
									vh_adj1 = circulator->first->vertex((circulator->second + 1) % 3);
									/*if (no == 0) std::cout << "vh_adj1	" << vh_adj1->point() << std::endl;*/
								}
							}
							edge_cir.push_back(circulator);
							++count;
						}  //9
					} while (++circulator != done); //8						
					//std::cout << "do-while" << std::endl;
					if ((CGAL::squared_distance(vh_center->point(), vh_adj->point()) < 3) && (CGAL::squared_distance(vh_center->point(), vh_adj1->point()) < 3))

					{
						edge_cir.clear();
						break;
					}
					if (vh_center == nullptr){ std::cout << "it is null" << std::endl; }
					if (vh_adj == nullptr){ std::cout << "it is null 1" << std::endl; }
					if (vh_adj1 == nullptr){ std::cout << "it is null 2" << std::endl; }

					double vec1 = ((vh_center->point().x() - vh_adj->point().x())*(vh_center->point().x() - vh_adj1->point().x()));
					double vec2 = ((vh_center->point().y() - vh_adj->point().y())*(vh_center->point().y() - vh_adj1->point().y()));

					double denom = (((CGAL::sqrt(((vh_center->point().x() - vh_adj->point().x())*(vh_center->point().x() - vh_adj->point().x())) + (vh_center->point().y() - vh_adj->point().y())*(vh_center->point().y() - vh_adj->point().y())))) * ((CGAL::sqrt(((vh_center->point().x() - vh_adj1->point().x())*(vh_center->point().x() - vh_adj1->point().x())) + (vh_center->point().y() - vh_adj1->point().y())*(vh_center->point().y() - vh_adj1->point().y())))));
					double ang;
					if (denom != 0)
						ang = std::acos((vec1 + vec2) / denom);

					//std::cout << "angle condition" << std::endl;
					if (ang<(175 * 3.14) / 180)
					{  //8 
						
						double dist = (CGAL::squared_distance(edge_cir.at(0)->first->vertex((edge_cir.at(0)->second + 1) % 3)->point(), edge_cir.at(0)->first->vertex((edge_cir.at(0)->second + 2) % 3)->point()));
						double dist1 = (CGAL::squared_distance(edge_cir.at(1)->first->vertex((edge_cir.at(1)->second + 1) % 3)->point(), edge_cir.at(1)->first->vertex((edge_cir.at(1)->second + 2) % 3)->point()));
						Edge_circulator circulator1;

						if (dist <= dist1){ circulator1 = edge_cir.at(1); }
						if (dist>dist1){ circulator1 = edge_cir.at(0); }

						if (circulator1 == NULL){ std::cout << "It is NULL " << std::endl; }
						CGAL::Object o = dt_sample.dual(circulator1);

						const Segment_2* s = CGAL::object_cast<Segment_2>(&o);
						const Ray_2* r = CGAL::object_cast<Ray_2>(&o);

						Segment_2* temp = new Segment_2;
						ThreshPoints.clear(); NewThreshPoints.clear(); Neighbors.clear(); Neighbor_Segments.clear();

						if (r)
						{  //10
							//std::cout<<"Ray"<<std::endl;
							//outputRays<<*r<<std::endl;
							if (tree.rootNode->rectangle.has_on_bounded_side((*r).source())){
								*temp = convToSeg(tree.rootNode->rectangle, *r);
							}
						}
						if (s)
						{  //10
							//std::cout<<"Seg"<<std::endl;
							//outputRays<<*s<<std::endl;
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
							Far_Point = get_Farthest_Point(*temp, Neighbor_Segments, vh_center->point());
							//++total_inGeometry;
							/*if (no == 0){
							for (fit f1 = dt_sample.finite_faces_begin(); f1 != dt_sample.finite_faces_end(); f1++)
							{
								for (int i = 0; i < 3; i++){
									Edge e = Edge(f1, i);
									//if(!dt.is_infinite(e))
									//{
									CGAL::Object o = dt_sample.dual(e);

									const Segment_2* s = CGAL::object_cast<Segment_2>(&o);
									const Ray_2* r = CGAL::object_cast<Ray_2>(&o);

									//int num_of_intersections=0;
									//Segment_2* temp = new Segment_2;
									//ThreshPoints.clear(); NewThreshPoints.clear();Neighbors.clear(); Neighbor_Segments.clear();				

									if (r)
									{
										std::cout << "Voronoi " << *r << std::endl;
									}
									if (s)
									{
										std::cout << "Voronoi " << *s << std::endl;
									}
									vh first = (f1->vertex(0));
									//unsigned int first_id = first->id;

									vh second = (f1->vertex(1));
									//unsigned int second_id = second->id;

									vh third = (f1->vertex(2));
									//unsigned int third_id = third->id;

									/*std::cout << "Delaunay Edges	" << std::endl;
									std::cout << first->point() << " "; std::cout << second->point() << std::endl;
									std::cout << second->point() << " "; std::cout << third->point() << std::endl;
									std::cout << third->point() << " "; std::cout << first->point() << std::endl;   
								}
							}*/  // For Paper

							/*std::vector<Segment_2> tempSeg = bBox(*temp, OT);
							std::cout << "Thresh Lines	:" << std::endl;
							//for (std::vector<Segment_2>::iterator th = Neighbor_Segments.begin(); th != Neighbor_Segments.end(); ++th)
							{
								//Segment_2 t1 = *th;
								std::cout << tempSeg.at(0).source() << " " << tempSeg.at(0).end() << std::endl;
								std::cout << tempSeg.at(1).source() << " " << tempSeg.at(1).end() << std::endl;
								std::cout << " Voronoi edge	" << tempSeg.at(2).source() << " " << tempSeg.at(2).end() << std::endl;
							}

							std::cout << "Reconstructed Curve (NN Crust)		:" << std::endl;
							//std::cout << "Voronoi Edge	" << *temp << "	Insertion Point	" << Far_Point << std::endl;
							for (std::vector<Segment_2>::iterator s_it = Neighbor_Segments.begin(); s_it != Neighbor_Segments.end(); ++s_it)
							{
								Segment_2 s1 = *s_it;
								std::cout << s1.source() << " " << s1.end() << std::endl;
							}*/   // For Paper
							
						//}
							//std::cout << "Inserted Point	" << Far_Point << std::endl;
							listing.insert(std::pair<Edge, Point_2>(*eit, Far_Point));
							//std::cin >> check;
							++no;
						}
						delete temp;
						break;
					}//8
					edge_cir.clear();
				}//7
			}//6
		}//5

		if (!listing.empty())
		{
			iterate = true;
			//flag = true;
			//int n = 0;
			//std::cout << ".....................................Geometry True............................." << std::endl;
			for (std::multimap<Edge, Point_2>::iterator m_it = listing.begin(); m_it != listing.end(); ++m_it)
			{
				vh vh1, vh2;
				vh1 = ((*m_it).first).first->vertex((((*m_it).first).second + 1) % 3);
				vh2 = ((*m_it).first).first->vertex((((*m_it).first).second + 2) % 3);
				if (dt_sample.is_edge(vh1, vh2))
				{
					vh tempVhandle = dt_sample.insert((*m_it).second);
					//if(outputRandomSample.is_open()){outputRandomSample<<(*m_it).second<<std::endl;}
					deFace(dt_sample, tempVhandle);
					markEdge(dt_sample, tree, tempVhandle);
				}
			}
			Topology(dt_sample, tree);
		}
	}	

	/*std::cout << "Restricted Edges after Geometric Approximation	" << std::endl;
	Edge_iterator et = dt_sample.finite_edges_begin();
	for (; et != dt_sample.finite_edges_end(); ++et)
	{  //5

		//std::cout << ".....................................Edge............................." << std::endl;
		if (et->first->correct_segments[et->second] == true)
		{  //6


			std::cout << et->first->vertex((et->second + 1) % 3)->point() << " " << et->first->vertex((et->second + 2) % 3)->point() << std::endl;


		}
	}*/ // For Paper

}