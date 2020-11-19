#include <iostream>
#include <utils.h>
#include <random>
#include <chrono>
#include <iomanip>

#include <gaussian.h>
#include <factor.h>
#include <variable.h>
#include <factor_graph.h>
std::mt19937 mt;

std::vector<std::pair<double, double>> make_random_poses(int N) {
    std::uniform_real_distribution<> dis(-10, 10);
    std::vector<std::pair<double, double>> poses;
    for (int i = 0; i < N; ++i) {
        if (i == 0) {
            /* Anchor first pose to the origin */
            poses.emplace_back(0, 0);
            continue;
        }
        double x = dis(mt);
        double y = dis(mt);
        poses.emplace_back(x, y);
    }
    return poses;
}

int main (int argc, char **argv) {
    if (argc != 2) {
        std::cout << "Expected number of poses as argument\n";
        exit(EXIT_FAILURE);
    }

    int N = atoi(argv[1]);
    mt.seed(1);
    Gaussian strong_prior = utils::make_prior(1e-8);
    Gaussian weak_prior = utils::make_prior(1e8);

    auto poses = make_random_poses(N);

    FactorGraph G;
    // Set all variable's initial estimate to (0, 0)
    // Anchor the nodes to the origin by applying strong prior to the first variable.
    for (size_t i = 0; i < poses.size(); ++i) {
        auto v = G.add_variable("v" + std::to_string(i));
        v->set_prior(i == 0 ? strong_prior : weak_prior);
    }

    std::uniform_int_distribution<> dis(0, poses.size() - 1);
    for (size_t i = 0; i < poses.size(); ++i) {
        // Make two connection with other node
        size_t j = dis(mt);
        while (i == j) { j = dis(mt); }
        size_t k = dis(mt);
        while (i == k || j == k) { k = dis(mt); }

        double x0 = poses[i].first;
        double y0 = poses[i].second;
        double x1 = poses[j].first;
        double y1 = poses[j].second;
        double x2 = poses[k].first;
        double y2 = poses[k].second;

        Factor *f1 = G.add_factor("f" + std::to_string(i) + "-" + std::to_string(j));
        f1->set_measurement(utils::make_measurement(x1 - x0, y1 - y0, 0.1, 0.1));
        G.connect(f1, {i, j});
        Factor *f2 = G.add_factor("f" + std::to_string(i) + "-" + std::to_string(k));
        f2->set_measurement(utils::make_measurement(x2 - x0, y2 - y0, 0.1, 0.1));
        G.connect(f2, {i, k});
    }


    auto start = std::chrono::high_resolution_clock::now();
    int num_iterations = 0;
    for (; num_iterations < 1000; ++num_iterations) {
        G.iteration();
        if (G.ARE() / N < 1e-7) { break; }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    // Verify the result
    bool success = true;
    for (size_t i = 0; i < poses.size(); ++i) {
        double dx = G.v(i)->belief().mu().x() - poses[i].first;
        double dy = G.v(i)->belief().mu().y() - poses[i].second;
        if (dx * dx + dy * dy > 1e-6) {
            std::cout << "Node " << i << " did not converge\n";
            std::cout << "Estimated x: " << G.v(i)->belief().mu().x() << ", Actual x:  " << poses[i].first << "\n";
            std::cout << "Estimated y: " << G.v(i)->belief().mu().y() << ", Actual y:  " << poses[i].second<< "\n";
            success = false;
        }
    }
    if (!success) {
        std::cout << "Test failed\n";
        exit(EXIT_FAILURE);
    }

    std::cout << N << " " << num_iterations << " " << std::setprecision(9) << diff.count() / num_iterations << " " << diff.count() << "\n";
    return 0;
}
