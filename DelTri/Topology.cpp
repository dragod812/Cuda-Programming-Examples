
#include "Topology.h"



void Topology(Delaunay& dt_sample, const QuadTree& tree)
{
	std::cout << ".....................................Topology............................." << std::endl;
	//listing.clear();
	//std::ofstream outputRays;
	//outputRays.open("OutputRays.txt", std::ofstream::out | std::ofstream::trunc);

	int check; int count = 0;
	bool iterate = true;
	while (iterate)
	{
		
		listing.clear();
		Edge_iterator eit = dt_sample.finite_edges_begin();
		for (; eit != dt_sample.finite_edges_end(); ++eit)
		{//2
			if (eit->first->correct_segments[eit->second] == false)
			{
				//std::cout << ".....................................Inside............................." << std::endl;
				//std::cout << eit->first->vertex((eit->second + 1) % 3)->point() << " " << eit->first->vertex((eit->second + 2) % 3)->point() << std::endl;
				iterate = false;
				
				CGAL::Object o = dt_sample.dual(eit); 

				const Segment_2* s = CGAL::object_cast<Segment_2>(&o);
				const Ray_2* r = CGAL::object_cast<Ray_2>(&o);

				int num_of_intersections = 0;
				Segment_2* temp = new Segment_2;
				ThreshPoints.clear(); NewThreshPoints.clear(); Neighbors.clear(); Neighbor_Segments.clear();

				if (r)
				{  
					//outputRays << *r << std::endl;
					if (tree.rootNode->rectangle.has_on_bounded_side((*r).source())){
						*temp = convToSeg(tree.rootNode->rectangle, *r);
					}
				}
				if (s)
				{	
					//outputRays << *s << std::endl;
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
					//std::cout << "num_of_intersections " << num_of_intersections << std::endl;
					
					if (num_of_intersections > 1)
					{
						Point_2 Far_Point = get_Farthest_Point(*temp, Neighbor_Segments, eit->first->vertex((eit->second + 1) % 3)->point());
						listing.insert(std::pair<Edge, Point_2>(*eit, Far_Point));
						/*if(count == 0){
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

									//std::cout << "Delaunay Edges	" << std::endl;
									//std::cout << first->point() << " "; std::cout << second->point() << std::endl;
									//std::cout << second->point() << " "; std::cout << third->point() << std::endl;
									//std::cout << third->point() << " "; std::cout << first->point() << std::endl;   // For Paper
								}
							}
							//std::cout << "Voronoi Edge	" << *temp << "	Insertion Point	" << Far_Point << std::endl;

							/*std::vector<Segment_2> tempSeg = bBox(*temp, OT);
							std::cout<<"Thresh Lines	:"<<std::endl;
							//for (std::vector<Segment_2>::iterator th = Neighbor_Segments.begin(); th != Neighbor_Segments.end(); ++th)
							{
								//Segment_2 t1 = *th;
								std::cout << tempSeg.at(0).source() << " " << tempSeg.at(0).end() << std::endl;
								std::cout << tempSeg.at(1).source() << " " << tempSeg.at(1).end() << std::endl;
								//std::cout << " Voronoi edge	" << tempSeg.at(2).source() << " " << tempSeg.at(2).end() << std::endl;
							}

							std::cout<<"Reconstructed Curve (NN Crust)		:"<<std::endl;
							for (std::vector<Segment_2>::iterator s_it = Neighbor_Segments.begin(); s_it != Neighbor_Segments.end(); ++s_it)
							{
								Segment_2 s1 = *s_it;
								std::cout << s1.source() << " " << s1.end() << std::endl;

							}
						}
						++count;*/ //For Paper
						//std::cin >> check;
					}  //5

					else if (num_of_intersections == 1)
					{
						//std::cout<<"Correct Edge	"<<eit->first->vertex((eit->second+1)%3)->point()<<" "<<eit->first->vertex((eit->second+2)%3)->point()<<std::endl;
						//std::cout << "Marked Edges" << std::endl;
						eit->first->correct_segments[eit->second] = true;
						fh opp_face = eit->first->neighbor(eit->second);
						int opp_index = opp_face->index(eit->first);
						opp_face->correct_segments[opp_index] = true;
					}
				}
				delete temp;
			}
		}//2

		//std::cin>>stop;
		if (!listing.empty())
		{
			//std::cin >> check;
			int n = 0;
			iterate = true;
			//std::cout << ".....................................True............................." << std::endl;
			for (std::multimap<Edge, Point_2>::iterator m_it = listing.begin(); m_it != listing.end(); ++m_it)
			{
				vh vh1, vh2;
				vh1 = ((*m_it).first).first->vertex((((*m_it).first).second + 1) % 3);
				vh2 = ((*m_it).first).first->vertex((((*m_it).first).second + 2) % 3);
				if ((dt_sample.is_edge(vh1, vh2)) && (((*m_it).first).first->correct_segments[((*m_it).first).second] == false))
				{
					vh tempVhandle = dt_sample.insert((*m_it).second);	
					//std::cout << "Point Inserted	" << tempVhandle->point() << std::endl;
					deFace(dt_sample, tempVhandle);
					markEdge(dt_sample, tree, tempVhandle);
				}
			}
			//std::cout<<"total_inEdgeMarking	"<<n<<std::endl;
		}
		
		if (iterate == false)
		{//3
			std::cout << ".....................................Manifold............................." << std::endl;
			//std::cin >> check;
			Point_2 Farthest_Point;
			eit = dt_sample.finite_edges_begin();
			for (; eit != dt_sample.finite_edges_end(); eit++)
			{  //4		
				if (eit->first->correct_segments[eit->second] == true)
				{  //5
					//std::cout<<eit->first->vertex((eit->second + 1) % 3)->point() << " " << eit->first->vertex((eit->second + 2) % 3)->point() << std::endl;
					for (int i = 1; i < 3; i++)
					{	//6
						int count = 0;
						vh vh_center = eit->first->vertex((eit->second + i) % 3);
						Edge_circulator circulator = dt_sample.incident_edges(vh_center), done(circulator);
						do
						{
							if (circulator->first->correct_segments[circulator->second] == true)
							{
								++count;
							}

						} while (++circulator != done);
						//std::cout<<"count is	"<<count<<std::endl;
						if ((count != 2))
						{  //7
							
							Farthest_Point = vh_center->point();
							Edge_circulator circulator1 = dt_sample.incident_edges(vh_center), done(circulator1);
							do
							{  //8
								if (circulator1->first->correct_segments[circulator1->second] == true)
								{  //9									
									CGAL::Object o = dt_sample.dual(circulator1);
									const Segment_2* s = CGAL::object_cast<Segment_2>(&o);
									const Ray_2* r = CGAL::object_cast<Ray_2>(&o);
									Point_2 Far_Point;
									int num_of_intersections = 0;
									Segment_2* temp = new Segment_2;
									ThreshPoints.clear(); NewThreshPoints.clear(); Neighbors.clear(); Neighbor_Segments.clear();

									if (r)
									{  //10
										//outputRays << *r << std::endl;
										if (tree.rootNode->rectangle.has_on_bounded_side((*r).source())){
											*temp = convToSeg(tree.rootNode->rectangle, *r);
										}
									}
									if (s)
									{  //10
										//outputRays << *s << std::endl;
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

										if ((CGAL::squared_distance(vh_center->point(), Far_Point)) > (CGAL::squared_distance(vh_center->point(), Farthest_Point)))
										{
											/*for (fit f1 = dt_sample.finite_faces_begin(); f1 != dt_sample.finite_faces_end(); f1++)
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

													std::cout << "Delaunay Edges	" << std::endl;
													std::cout << first->point() << " "; std::cout << second->point() << std::endl;
													std::cout << second->point() << " "; std::cout << third->point() << std::endl;
													std::cout << third->point() << " "; std::cout << first->point() << std::endl;
												}
											}
											//std::cout << "Voronoi Edge	" << *temp << "	Insertion Point	" << Far_Point << std::endl;
											for (std::vector<Segment_2>::iterator s_it = Neighbor_Segments.begin(); s_it != Neighbor_Segments.end(); ++s_it)
											{
												Segment_2 s1 = *s_it;
												std::cout << s1.source() << " " << s1.end() << std::endl;;

											}*/   //For Paper
											
											Farthest_Point = Far_Point;
											//std::cout << "Inserted Point in Manifold	" << Farthest_Point << std::endl;
											listing.insert(std::pair<Edge, Point_2>(*eit, Farthest_Point));
											//std::cin >> check;
										}
									}
									delete temp;
								}
							} while (++circulator1 != done);//8							
							break;
						}//7
					}//6						
				}//5

			}//4
			if (!listing.empty())
			{
				int n = 0;
				iterate = true;
				for (std::multimap<Edge, Point_2>::iterator m_it = listing.begin(); m_it != listing.end(); ++m_it)
				{
					vh vh1, vh2;
					vh1 = ((*m_it).first).first->vertex((((*m_it).first).second + 1) % 3);
					vh2 = ((*m_it).first).first->vertex((((*m_it).first).second + 2) % 3);
					if (dt_sample.is_edge(vh1, vh2))
					{
						vh tempVhandle = dt_sample.insert((*m_it).second);
						//if(outputRandomSample.is_open()){outputRandomSample<<(*m_it).second<<std::endl;}
						//std::cout << "Point Inserted	" << tempVhandle->point() << std::endl;
						deFace(dt_sample, tempVhandle);
						markEdge(dt_sample, tree, tempVhandle);
					}
				}
			}
		}//3
	}
}