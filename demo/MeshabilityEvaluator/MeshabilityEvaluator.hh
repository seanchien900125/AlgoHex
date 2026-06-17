#pragma once

#include <cmath>
#include <iomanip>
#include <limits>
#include <string>
#include <vector>
#include <AlgoHex/Util/json.hh>
#include <AlgoHex/TypeDef.hh>
#include <AlgoHex/Args.hh>
#include <AlgoHex/SmoothOctahedralFieldGeneratorT.hh>
#include <AlgoHex/SmoothOctahedralFieldGeneratorCellBasedT.hh>
#include <AlgoHex/FieldConstraintGenerator.hh>
#include <AlgoHex/SingularGraphExtractionT.hh>
#include <AlgoHex/Parametrization3DT.hh>
#include <AlgoHex/FrameFieldOptimizer3DT.hh>
#include <AlgoHex/Config/Export.hh>

#include <AlgoHex/LocallyMeshableField/LocallyMeshableFieldGenerationT.hh>
#include <AlgoHex/LocallyMeshableField/MeshProperties.hh>
#include <AlgoHex/LocallyMeshableField/OneringParameterizationChecker.hh>
#include <AlgoHex/LocallyMeshableField/LocalMeshabilityCheckerWithFrames.hh>
#include <AlgoHex/LocallyMeshableField/LocalMeshabilityChecker.hh>

#include <AlgoHex/LocallyMeshableField/Comparisons/SplitPaperT.hh>
#include <AlgoHex/LocallyMeshableField/Comparisons/CollapsePaperT.hh>

//== NAMESPACES ===============================================================

namespace AlgoHex
{
template<class MeshT>
class MeshabilityCheckerOutputJson
{
public:
    MeshabilityCheckerOutputJson(MeshT &_mesh)
                    : mesh_(_mesh),
                      local_meshability_checker_(_mesh) { }

    ~MeshabilityCheckerOutputJson() = default;

    nlohmann::json &json_data() { return json_data_; }

    double check_local_meshability(const bool _align_to_sge = false)
    {
        Debug::ScopedOutputLevel sol(0);

        std::cout << "##### Check local meshability ..." << std::endl;
        int n(0.0);
        int n_meshable(0.0);

        int n_on_feature_vertex(0);
        int n_on_feature_vertex_meshable(0);
        int n_on_feature_edge(0);
        int n_on_feature_edge_meshable(0);
        int n_on_feature_face(0);
        int n_on_feature_face_meshable(0);

        int n_on_singular_vertex(0);
        int n_on_singular_vertex_meshable(0);
        int n_on_singular_node(0);
        int n_on_singular_node_meshable(0);
        int n_zipper_node(0);
        int n_zipper_node_meshable(0);
        int n_on_singular_arc(0);
        int n_on_singular_arc_meshable(0);

        for (VIt v_it = mesh_.v_iter(); v_it.valid(); ++v_it)
        {
            ++n;
            std::cerr << "check vh " << *v_it << " ";
            bool ilm = lmcwf_.is_locally_meshable(*v_it, _align_to_sge, 0, false);

            if (ilm)
            {
            ++n_meshable;
            }
            else
            {
            std::cerr << "vertex " << *v_it << " is not locally meshable." << std::endl;
            }

            if (is_on_feature_vertex(*v_it))
            {
            ++n_on_feature_vertex;
            if (ilm)
                ++n_on_feature_vertex_meshable;
            }

            if (is_on_feature_edge(*v_it))
            {
            ++n_on_feature_edge;
            if (ilm)
                ++n_on_feature_edge_meshable;
            }

            if (is_on_feature_face(*v_it))
            {
            ++n_on_feature_face;
            if (ilm)
                ++n_on_feature_face_meshable;
            }

            if (is_singular_vertex(*v_it))
            {
            ++n_on_singular_vertex;
            if (ilm)
                ++n_on_singular_vertex_meshable;
            }

            if (is_on_singular_node(*v_it))
            {
            ++n_on_singular_node;
            if (ilm)
                ++n_on_singular_node_meshable;
            }

            if (is_zipper_node(*v_it))
            {
            ++n_zipper_node;
            if (ilm)
                ++n_zipper_node_meshable;
            }

            if (is_on_singular_arc(*v_it))
            {
            ++n_on_singular_arc;
            if (ilm)
                ++n_on_singular_arc_meshable;
            }
        }
        std::cerr << std::endl;

        double feature_vertex_ratio = double(n_on_feature_vertex_meshable) / double(n_on_feature_vertex);
        if (!std::isfinite(feature_vertex_ratio))
            feature_vertex_ratio = 1.0;

        std::cerr << "#vertices          = " << std::setw(5) << n << ", #meshable = " << std::setw(5) << n_meshable << " ("
            << double(n_meshable) / double(n) * 100.0 << "%)\n";
        std::cerr << "#on feature node = " << std::setw(5) << n_on_feature_vertex << ", #meshable = " << std::setw(5)
            << n_on_feature_vertex_meshable << " (" << feature_vertex_ratio * 100.0 << "%)\n";
        std::cerr << "#on feature edge   = " << std::setw(5) << n_on_feature_edge << ", #meshable = " << std::setw(5)
            << n_on_feature_edge_meshable << " ("
            << double(n_on_feature_edge_meshable) / double(n_on_feature_edge) * 100.0 << "%)\n";
        std::cerr << "#on feature face   = " << std::setw(5) << n_on_feature_face << ", #meshable = " << std::setw(5)
            << n_on_feature_face_meshable << " ("
            << double(n_on_feature_face_meshable) / double(n_on_feature_face) * 100.0 << "%)\n";
        std::cerr << "#on singular node  = " << std::setw(5) << n_on_singular_node << ", #meshable = " << std::setw(5)
            << n_on_singular_node_meshable << " ("
            << double(n_on_singular_node_meshable) / double(n_on_singular_node) * 100.0 << "%)\n";
        std::cerr << "#turning point  = " << std::setw(5) << n_zipper_node << ", #meshable = " << std::setw(5)
            << n_zipper_node_meshable << " (" << double(n_zipper_node_meshable) / double(n_zipper_node) * 100.0
            << "%)\n";
        std::cerr << "#on singular arc   = " << std::setw(5) << n_on_singular_arc << ", #meshable = " << std::setw(5)
            << n_on_singular_arc_meshable << " ("
            << double(n_on_singular_arc_meshable) / double(n_on_singular_arc) * 100.0 << "%)\n";
        std::cerr << "#singular vertices   = " << std::setw(5) << n_on_singular_vertex << ", #meshable = " << std::setw(5)
            << n_on_singular_vertex_meshable << " ("
            << double(n_on_singular_vertex_meshable) / double(n_on_singular_vertex) * 100.0 << "%)\n";
        std::cerr << "#feature vertices   = " << std::setw(5) << n_on_feature_edge << ", #meshable = " << std::setw(5)
            << n_on_feature_edge_meshable << " ("
            << double(n_on_feature_edge_meshable) / double(n_on_feature_edge) * 100.0 << "%)\n";
        std::cerr << "#len singular edge   = " << std::setw(5) << len_singular_edges() << "\n";
        std::cerr << "#len complex singular edge   = " << std::setw(5) << len_complex_singular_edges() << "\n";

        json_data_.clear();
        json_data_["percentage_meshable_vertices"] = double(n_meshable) / double(n);
        json_data_["non-meshable"] = n - n_meshable;
        json_data_["all vertices"] = n;
        json_data_["singular nodes"] = n_on_singular_node - n_on_singular_node_meshable;
        json_data_["turning points"] = n_zipper_node;
        json_data_["singular vertices"] = n_on_singular_vertex - n_on_singular_vertex_meshable;
        json_data_["feature vertices"] = n_on_feature_edge - n_on_feature_edge_meshable;
        json_data_["len complex singular edges"] = len_complex_singular_edges();
        return double(n_meshable) / double(n);
    }

    double check_edge_local_meshability(const bool _align_to_sge = false)
    {
        double percentage_edge_meshability = local_meshability_checker_.check_edge_local_meshability(_align_to_sge);
        auto lmc_json_data = local_meshability_checker_.json_data();
        json_data_["percentage_meshable_edges"] = percentage_edge_meshability;
        json_data_["n complex singular edges"] = lmc_json_data.at("n complex singular edges");
        json_data_["n singular edges"] = lmc_json_data.at("n singular edges");
        return percentage_edge_meshability;
    }

    bool is_on_feature_vertex(const VH _vh) const
    {
        return local_meshability_checker_.is_on_feature_vertex(_vh);
    }

    bool is_on_feature_edge(const VH _vh) const
    {
        return local_meshability_checker_.is_on_feature_edge(_vh);
    }

    bool is_on_feature_face(const VH _vh) const
    {
        return local_meshability_checker_.is_on_feature_face(_vh);
    }

    bool is_on_singular_node(const VH _vh) const
    {
        return local_meshability_checker_.is_on_singular_node(_vh);
    }

    bool is_zipper_node(const VH _vh) const
    {
        return local_meshability_checker_.is_zipper_node(_vh);
    }

    bool is_on_singular_arc(const VH _vh) const
    {
        return local_meshability_checker_.is_on_singular_arc(_vh);
    }

    bool is_singular_vertex(const VH _vh) const
    {
        return local_meshability_checker_.is_singular_vertex(_vh);
    }

    int n_complex_singular_edges() const
    {
        return local_meshability_checker_.n_complex_singular_edges();
    }

    int n_singular_edges() const
    {
        return local_meshability_checker_.n_singular_edges();
    }

    double len_complex_singular_edges() const
    {
        return local_meshability_checker_.len_complex_singular_edges();
    }

    double len_singular_edges() const
    {
        return local_meshability_checker_.len_singular_edges();
    }

    bool &verbose() { return local_meshability_checker_.verbose(); }

private:
    MeshT &mesh_;
    LocalMeshabilityChecker<MeshT> local_meshability_checker_;
    LocalMeshabilityCheckerWithFrames<MeshT> lmcwf_{mesh_};
    nlohmann::json json_data_;
};

class ALGOHEX_EXPORT MeshabilityEvaluator
{
public:
    MeshabilityEvaluator(const Args &args) : args_(args) {}
    
    template<class MeshT>
    int evaluate(MeshT &tetmesh);

private:
    const Args &args_;

    template<class MeshT>
    void check_local_meshability(MeshT &tetmesh, nlohmann::json &json_data);

    template<class MeshT>
    void get_initial_frame_field(MeshT &tetmesh, nlohmann::json &json_data);

    template<class MeshT>
    std::vector<EH> get_feature_edges(MeshT &tetmesh);

    void evaluate_meshability_from_json(const nlohmann::json &json_data);

};


class RuleBasedScoreEvaluator
{
public:
    RuleBasedScoreEvaluator() = default;

    double evaluate(const nlohmann::json &json_data)
    {
        return 0.0;
    }

private: 
    double threshold_ = 0.5;
};

template<class MeshT>
void load_tetmesh(const std::string &inFileName, MeshT &tetmesh);

//=============================================================================
} // namespace AlgoHex
//=============================================================================
