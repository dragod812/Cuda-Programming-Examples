std::vector<Point_2> insidePoints(const QuadTree& tree, const Segment_2& edge, const double& thresh)
{
	//std::cout.precision(17);
	std::vector<std::vector<Point_2> > tempPoints;
	//tempPoints.reserve(5);
	std::queue<std::shared_ptr<Node>> nodeQueue;
	//std::unordered_map<Point_2,int> mapping;
	//int count=0;

	std::vector<Point_2> th_Points;
	//th_Points.reserve(10);
	std::shared_ptr<Node> temp;
	//temp.reserve(4);

	//std::cout<<"insidePoint"<<std::endl;
	std::vector<Segment_2> tempSeg = bBox(edge, thresh);
	//std::cout<<"bBox"<<std::endl;

	if (tree.rootNode == nullptr){ return th_Points; }
	else
	{
		nodeQueue.push(tree.rootNode);
		while (!nodeQueue.empty()){
			temp = nodeQueue.front();

			if (temp->child.size() == 0)
			{
				tempPoints.push_back(temp->insidePoints);
			}

			for (int i = 0; i<temp->child.size(); ++i)
			{
				bool boxInside = false;
				if ((CGAL::squared_distance(tempSeg.at(2), temp->child.at(i)->rectangle.vertex(0)) <= thresh) 
				|| (CGAL::squared_distance(tempSeg.at(2), temp->child.at(i)->rectangle.vertex(1)) <= thresh) 
				|| (CGAL::squared_distance(tempSeg.at(2), temp->child.at(i)->rectangle.vertex(2)) <= thresh) 
				|| (CGAL::squared_distance(tempSeg.at(2), temp->child.at(i)->rectangle.vertex(3)) <= thresh))
				{
					boxInside = true;
				}

				
				if ((boxInside == true) || (intersect(tempSeg.at(0), temp->child.at(i)->rectangle)) || (intersect(tempSeg.at(1), temp->child.at(i)->rectangle)) || (intersect(tempSeg.at(2), temp->child.at(i)->rectangle)))
				{
					nodeQueue.push(temp->child.at(i));
				}
			}
			nodeQueue.pop();
		}
	}
	for (int i = 0; i<tempPoints.size(); ++i){
		//std::cout<<"2"<<std::endl;
		for (int j = 0; j<tempPoints[i].size(); ++j){
			thresh_Points(edge, tempPoints[i].at(j), thresh, th_Points);
		}
	}
	//std::vector<Point_2> th1_Points;
	//th1_Points.reserve(10);
	//std::cout<<"done"<<std::endl;
	return th_Points;
}
