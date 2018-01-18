template<class m1, class m2>
std::map<m1, m2> loadMap(cv::FileNode n, std::string s)
{
    std::map<m1, m2> m;

    return m;
}

template<class c1>
std::vector<c1> loadVector(cv::FileNode n, std::string s)
{
    std::vector<c1> v;

    v = (std::vector<c1>) n[s];

    return v;
}

cv::Mat loadMat(cv::FileNode n, std::string s)
{
    cv::Mat m;
    n[s] >> m;
    return m;
}
