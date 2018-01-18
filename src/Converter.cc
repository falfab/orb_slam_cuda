/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

#include "Converter.h"
#include <algorithm>

namespace ORB_SLAM2
{

std::vector<cv::Mat> Converter::toDescriptorVector(const cv::Mat &Descriptors)
{
    std::vector<cv::Mat> vDesc;
    vDesc.reserve(Descriptors.rows);
    for (int j = 0; j < Descriptors.rows; j++)
        vDesc.push_back(Descriptors.row(j));

    return vDesc;
}

g2o::SE3Quat Converter::toSE3Quat(const cv::Mat &cvT)
{
    Eigen::Matrix<double, 3, 3> R;
    R << cvT.at<float>(0, 0), cvT.at<float>(0, 1), cvT.at<float>(0, 2),
        cvT.at<float>(1, 0), cvT.at<float>(1, 1), cvT.at<float>(1, 2),
        cvT.at<float>(2, 0), cvT.at<float>(2, 1), cvT.at<float>(2, 2);

    Eigen::Matrix<double, 3, 1> t(cvT.at<float>(0, 3), cvT.at<float>(1, 3), cvT.at<float>(2, 3));

    return g2o::SE3Quat(R, t);
}

cv::Mat Converter::toCvMat(const g2o::SE3Quat &SE3)
{
    Eigen::Matrix<double, 4, 4> eigMat = SE3.to_homogeneous_matrix();
    return toCvMat(eigMat);
}

cv::Mat Converter::toCvMat(const g2o::Sim3 &Sim3)
{
    Eigen::Matrix3d eigR = Sim3.rotation().toRotationMatrix();
    Eigen::Vector3d eigt = Sim3.translation();
    double s = Sim3.scale();
    return toCvSE3(s * eigR, eigt);
}

cv::Mat Converter::toCvMat(const Eigen::Matrix<double, 4, 4> &m)
{
    cv::Mat cvMat(4, 4, CV_32F);
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            cvMat.at<float>(i, j) = m(i, j);

    return cvMat.clone();
}

cv::Mat Converter::toCvMat(const Eigen::Matrix3d &m)
{
    cv::Mat cvMat(3, 3, CV_32F);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            cvMat.at<float>(i, j) = m(i, j);

    return cvMat.clone();
}

cv::Mat Converter::toCvMat(const Eigen::Matrix<double, 3, 1> &m)
{
    cv::Mat cvMat(3, 1, CV_32F);
    for (int i = 0; i < 3; i++)
        cvMat.at<float>(i) = m(i);

    return cvMat.clone();
}

cv::Mat Converter::toCvSE3(const Eigen::Matrix<double, 3, 3> &R, const Eigen::Matrix<double, 3, 1> &t)
{
    cv::Mat cvMat = cv::Mat::eye(4, 4, CV_32F);
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            cvMat.at<float>(i, j) = R(i, j);
        }
    }
    for (int i = 0; i < 3; i++)
    {
        cvMat.at<float>(i, 3) = t(i);
    }

    return cvMat.clone();
}

Eigen::Matrix<double, 3, 1> Converter::toVector3d(const cv::Mat &cvVector)
{
    Eigen::Matrix<double, 3, 1> v;
    v << cvVector.at<float>(0), cvVector.at<float>(1), cvVector.at<float>(2);

    return v;
}

Eigen::Matrix<double, 3, 1> Converter::toVector3d(const cv::Point3f &cvPoint)
{
    Eigen::Matrix<double, 3, 1> v;
    v << cvPoint.x, cvPoint.y, cvPoint.z;

    return v;
}

Eigen::Matrix<double, 3, 3> Converter::toMatrix3d(const cv::Mat &cvMat3)
{
    Eigen::Matrix<double, 3, 3> M;

    M << cvMat3.at<float>(0, 0), cvMat3.at<float>(0, 1), cvMat3.at<float>(0, 2),
        cvMat3.at<float>(1, 0), cvMat3.at<float>(1, 1), cvMat3.at<float>(1, 2),
        cvMat3.at<float>(2, 0), cvMat3.at<float>(2, 1), cvMat3.at<float>(2, 2);

    return M;
}

std::vector<float> Converter::toQuaternion(const cv::Mat &M)
{
    Eigen::Matrix<double, 3, 3> eigMat = toMatrix3d(M);
    Eigen::Quaterniond q(eigMat);

    std::vector<float> v(4);
    v[0] = q.x();
    v[1] = q.y();
    v[2] = q.z();
    v[3] = q.w();

    return v;
}

// PBA conversion methods
std::vector<std::vector<float>> Converter::toPbaDataMatrix(vector<KeyFrame *> &vpKFs, vector<MapPoint *> &vpMP, vector<MapPoint *> &mpOrdered)
{
    int num_observation = 0;

    // Utilizzo vector al posto di cv::Mat per migliore gestione della memoria.
    vector<vector<float>> pVobs;  // matrice delle observations
    vector<vector<float>> pVcops; // matrice delle opzioni camera
    int kfBadCount = 0;
    int mpBadCount = 0;

    //////////////////////////////////////////////////////////////////
    // VALORI CAMERE //
    // per ogni frame (ad ogni frame corrisponde una camera)
    for (size_t i = 0; i < vpKFs.size(); i++)
    {
        KeyFrame *pKf = vpKFs[i];
        if (pKf->isBad())
        {
            kfBadCount++;
            continue;
        }

        cv::Mat pRm = pKf->GetRotation();
        cv::Mat pTv = pKf->GetTranslation();

        cv::Mat pRv = cv::Mat::zeros(1, 3, CV_64F);
        cv::Rodrigues(pRm, pRv);

        vector<float> pVcop = {pRv.at<float>(0, 0),
                               pRv.at<float>(0, 1),
                               pRv.at<float>(0, 2),
                               pTv.at<float>(0, 0),
                               pTv.at<float>(0, 1),
                               pTv.at<float>(0, 2),
                               pKf->mK.at<float>(0, 0),
                               0, 0};
        pVcops.push_back(pVcop);

        // VALORI OSSERVAZIONI //
        // per ogni map point associato al keyframe
        for (auto mp : pKf->GetMapPoints())
        {
            if (mp->isBad())
            {
                mpBadCount++;
                continue;
            }

            // VALORI MAP POINT //
            int mpPos;

            //se il punto non e' contenuto nel vector, lo aggiungo in coda
            auto mpIterator = std::find(mpOrdered.begin(), mpOrdered.end(), mp);
            if (mpIterator == mpOrdered.end())
            {
                mpOrdered.push_back(mp);
                mpPos = mpOrdered.size() - 1; //la posizione e' l'ultima del vector
            }
            else
            {
                //ricavo la posizione da quella data dalla funzione find()
                mpPos = std::distance(mpOrdered.begin(), mpIterator);
            }

            cv::Mat pMwp = mp->GetWorldPos();

            // P = R * X + t        (conversione fra coordinate mondo a coordinate camera)
            cv::Mat P = pRm * pMwp + pTv;
            // p = -P / P.z         (normalizzazione)
            cv::Mat p = -P / P.at<float>(2, 0);
            float focal_length = pKf->mK.at<float>(0, 0);
            cv::Mat pf = focal_length * p;
            num_observation++;
            vector<float> pVob = {i, mpPos, pf.at<float>(0, 0), pf.at<float>(1, 0)};
            pVobs.push_back(pVob);
        }
    }

    // the return matrix
    std::vector<std::vector<float>> vPba;

    // <num_cameras> <num_points> <num_observations>
    std::vector<float> vParams = {vpKFs.size(), mpOrdered.size(), num_observation};

    vPba.push_back(vParams);

    // <camera_index_1> <point_index_1> <x_1> <y_1>
    for (vector<float> vf : pVobs)
    {
        vPba.push_back(vf);
    }
    // <camera_1> ... <camera_n>
    for (vector<float> vf : pVcops)
    {
        vPba.push_back(vf);
    }
    // <point_1> ... <point_n>
    for (MapPoint *mp : mpOrdered)
    {
        cv::Mat coord = mp->GetWorldPos();
        std::vector<float> vMp;
        for (int i = 0; i < 3; i++)
        {
            vMp.push_back(coord.at<float>(i, 0));
        }
        vPba.push_back(vMp);
    }
    return vPba;
}

// Come il metodo sopra, pero ordina i dati per l'id del map point
std::vector<std::vector<float>> Converter::toPbaDataMatrixMpOrder(vector<KeyFrame *> &vpKFs, vector<MapPoint *> &vpMP)
{
    // Utilizzo vector al posto di cv::Mat per migliore gestione della memoria.
    vector<vector<float>> pVobs;  // matrice delle observations
    vector<vector<float>> pVcops; // matrice delle opzioni camera
    int kfBadCount = 0;
    int mpBadCount = 0;

    // VALORI CAMERE //
    // per ogni frame (ad ogni frame corrisponde una camera)
    for (size_t i = 0; i < vpKFs.size(); i++)
    {
        KeyFrame *pKf = vpKFs[i];
        if (pKf->isBad())
        {
            kfBadCount++;
            continue;
        }

        cv::Mat pRm = pKf->GetRotation();
        cv::Mat pTv = pKf->GetTranslation();

        cv::Mat pRv = cv::Mat::zeros(1, 3, CV_64F);
        cv::Rodrigues(pRm, pRv);

        vector<float> pVcop = {pRv.at<float>(0, 0),
                               pRv.at<float>(0, 1),
                               pRv.at<float>(0, 2),
                               pTv.at<float>(0, 0),
                               pTv.at<float>(0, 1),
                               pTv.at<float>(0, 2),
                               pKf->mK.at<float>(0, 0),
                               0, 0};
        pVcops.push_back(pVcop);
    }

    // VALORI OSSERVAZIONI //
    //osservazioni già disponibili a quanto pare...
    for (size_t i = 0; i < vpMP.size(); i++)
    {
        MapPoint *pMP = vpMP[i];
        if (pMP->isBad())
        {
            mpBadCount++;
            continue;
        }

        const map<KeyFrame *, size_t> observations = pMP->GetObservations();

        for (map<KeyFrame *, size_t>::const_iterator mit = observations.begin(); mit != observations.end(); mit++)
        {
            KeyFrame *pKF = mit->first;
            if (pKF->isBad())
                continue;

            //creazione delle osservazioni con le formule fornite da PBA
            cv::Mat pRm = pKF->GetRotation();
            cv::Mat pTv = pKF->GetTranslation();
            cv::Mat pMwp = pMP->GetWorldPos();
            // P = R * X + t        (conversione fra coordinate mondo a coordinate camera)
            cv::Mat P = pRm * pMwp + pTv;
            // p = -P / P.z         (normalizzazione)
            cv::Mat p = -P / P.at<float>(2, 0);
            float focal_length = pKF->mK.at<float>(0, 0);
            cv::Mat pf = focal_length * p;

            //idea: le camere sono meno dei punti, conviene scorrere queste
            for (size_t j = 0; j < vpKFs.size(); j++)
            {
                KeyFrame *kf = vpKFs[j];
                if (kf->mnId == pKF->mnId)
                {
                    vector<float> pVob = {j, i, pf.at<float>(0, 0), pf.at<float>(1, 0)};
                    pVobs.push_back(pVob);
                    break;
                }
            }
        }
    }

    // the return matrix
    std::vector<std::vector<float>> vPba;

    // <num_cameras> <num_points> <num_observations>
    std::vector<float> vParams = {vpKFs.size(), vpMP.size(), pVobs.size()};

    vPba.push_back(vParams);

    // <camera_index_1> <point_index_1> <x_1> <y_1>
    for (vector<float> vf : pVobs)
    {
        vPba.push_back(vf);
    }
    // <camera_1> ... <camera_n>
    for (vector<float> vf : pVcops)
    {
        vPba.push_back(vf);
    }
    // <point_1> ... <point_n>
    for (MapPoint *mp : vpMP)
    {
        cv::Mat coord = mp->GetWorldPos();
        std::vector<float> vMp;
        for (int i = 0; i < 3; i++)
        {
            vMp.push_back(coord.at<float>(i, 0));
        }
        vPba.push_back(vMp);
    }

    return vPba;
}

void Converter::dataFromPbaFile(char file_name[], vector<KeyFrame *> &vpKFs, vector<MapPoint *> &vpMP, vector<MapPoint *> &mpOrdered)
{
    std::ifstream file(file_name, std::ios_base::app); //apro lo stream sul file da leggere
    if (file.is_open())
    {
        //lettura della prima riga di intestazione
        int numKF, numMP, numOBS;
        file >> numKF >> numMP >> numOBS; //leggo la prima riga di intestazione
        std::cout << numKF << " " << numMP << " " << numOBS << std::endl;

        //lettura Observations
        int i = 1;
        while (i <= numOBS)
        {
            int idKF, idMP;
            float xn, yn;
            file >> idKF >> idMP >> xn >> yn;
            i++;
        }

        //lettura KeyFrames
        i = 1;
        while (i <= numKF)
        {
            float r1, r2, r3;
            float t1, t2, t3;
            float f, k1, k2;
            file >> r1 >> r2 >> r3 >> t1 >> t2 >> t3 >> f >> k1 >> k2;
            // std::cout << r1 << " " << r2 << " " << r3 << " " << t1 << " " << t2 << " " << t3 << " " << f << " " << k1 << " " << k2 << std::endl;

            cv::Mat rotVec = (cv::Mat_<float>(3, 1) << t1, t2, t3); //salvo le rotazioni in un vettore
            cv::Mat rotMat = cv::Mat::zeros(4, 4, CV_64F);
            cv::Rodrigues(rotVec, rotMat); //converto il vettore delle rotazioni in matrice 3x3

            //nei KeyFrame è possibile impostare solo la matrice 4x4 della posa della camera, formata così:
            //[rotazione|traslazione]
            //[---------|-----------]
            //[  0 0 0  |     1     ]

            cv::Mat poseMat = (cv::Mat_<float>(4, 4) << rotMat.at<float>(0, 0), rotMat.at<float>(0, 1), rotMat.at<float>(0, 2), t1,
                               rotMat.at<float>(1, 0), rotMat.at<float>(1, 1), rotMat.at<float>(1, 2), t2,
                               rotMat.at<float>(2, 0), rotMat.at<float>(2, 1), rotMat.at<float>(2, 2), t3,
                               0, 0, 0, 1);

            //ci sono piccole differenze nella stampa usando i due metodi qui sotto... per arrotondamenti diversi
            // std::cout << "poseMat: " << poseMat << std::endl;
            // std::cout << "poseMat(0, 0): " << poseMat.at<float>(0, 0) << std::endl;
            // std::cout << "poseMat(1, 3): " << poseMat.at<float>(1, 3) << std::endl;
            // std::cout << "poseMat(3, 3): " << poseMat.at<float>(3, 3) << std::endl;
            i++;
        }

        //lettura MapPoints
        i = 1;
        while (i <= numMP)
        {
            float x, y, z;
            file >> x >> y >> z;
            std::cout << x << " " << y << " " << z << std::endl;
            i++;
        }
        file.close();
    }
    else
        cout << "Unable to open file";
}

void Converter::printPbaMatrixToFile(std::vector<std::vector<float>> &vPba, char file_name[])
{
    std::ofstream file;
    file.open(file_name, std::ios_base::app);

    for (auto line : vPba)
    {
        for (auto element : line)
        {
            file << element << " ";
        }
        file << std::endl;
    }
    file.close();
}

} //namespace ORB_SLAM
