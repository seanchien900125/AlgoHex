
#include <OpenVolumeMesh/FileManager/FileManager.hh>
#include <OpenVolumeMesh/FileManager/VtkColorReader.hh>
#include <OpenVolumeMesh/IO/WriteOptions.hh>
#include <OpenVolumeMesh/IO/ReadOptions.hh>
#include <OpenVolumeMesh/IO/ovmb_write.hh>
#include <OpenVolumeMesh/IO/ovmb_read.hh>
#include <AlgoHex/CommonOVMBDecoders/PropertyCodecsSHCoeffs.hh>
#include "MeshabilityEvaluator.hh"

namespace AlgoHex
{
using Point = OpenVolumeMesh::Geometry::Vec3d;
using TetMesh = OVM::TetrahedralGeometryKernel<Point, OVM::TetrahedralMeshTopologyKernel>;

template<typename MeshT>
void read_ovmb_file(const std::string &filename, MeshT &mesh)
{
  OVM::IO::ReadOptions ro;
  OVM::IO::PropertyCodecs propCodecs = OVM::IO::g_default_property_codecs;
  OVM::IO::register_eigen_codecs(propCodecs);
  OVM::IO::register_algohex_codecs(propCodecs);

  std::ifstream iff(filename.c_str(), std::ios::binary);
  OVM::IO::ovmb_read(iff, mesh, ro, propCodecs);
  iff.close();
}

template<class MeshT>
void initialize_feature_properties(const Args &args, MeshT &tetmesh, nlohmann::json &json_data)
{
  std::cout << "No prescribed feature tags found. Detecting features from dihedral angle threshold " << args.dihedral_angle << std::endl;
  auto feature_fprop = tetmesh.template request_face_property<int>("AlgoHex::FeatureFaces", 0);
  for (const auto fhi: tetmesh.faces())
    if (tetmesh.is_boundary(fhi))
      feature_fprop[fhi] = 1;
    else
      feature_fprop[fhi] = 0;
  tetmesh.set_persistent(feature_fprop, true);

  AlgoHex::FaceNormalCache normal_cache{tetmesh};
  AlgoHex::DihedralFeatureDetector feature_detector{tetmesh, normal_cache};
  auto feature_edges = feature_detector.compute(args.dihedral_angle);
  auto dihedral_angles = feature_detector.compute_dihedral_angles();
  double min_dihedral_angle = *std::min_element(dihedral_angles.begin(), dihedral_angles.end());

  auto feature_edge_prop = tetmesh.template request_edge_property<int>("AlgoHex::FeatureEdges", 0);
  tetmesh.set_persistent(feature_edge_prop, true);
  for (const auto eh: feature_edges)
    feature_edge_prop[eh] = 1;

  MeshPropertiesT<MeshT> mp(tetmesh);
  mp.initialize_feature_vertex_property();

  json_data["NumberFeatureEdges"] = feature_edges.size();
  json_data["MinimumDihedralAngle"] = min_dihedral_angle;
}

template<class MeshT>
void load_tetmesh(const std::string &inFileName, MeshT &tetmesh)
{
  if (!inFileName.empty())
  {
    //identify file format
    auto found = inFileName.find_last_of(".");
    auto file_type = inFileName.substr(found + 1);

    if (file_type == "vtk")
    {
      OpenVolumeMesh::Reader::VtkColorReader vtkfm;
      vtkfm.readFile(inFileName, tetmesh, true, true);
    }
    else if (file_type == "ovm")
    {
      OpenVolumeMesh::IO::FileManager fm;
      fm.readFile(inFileName, tetmesh);
    } else if (file_type == "ovmb")
    {
      read_ovmb_file(inFileName, tetmesh);
    }
    else
    {
      std::cerr << "Error: the file type is not supported!" << std::endl;
      return;
    }
  }
}

template<class MeshT>
int MeshabilityEvaluator::evaluate(MeshT &tetmesh)
{
  // new a json type to store the output
  nlohmann::json json_data;

  if (this->args_.inFileName.empty())
  {
    std::cerr << "No input is given!" << std::endl;
    return -1;
  }
  get_initial_frame_field(tetmesh, json_data);
  check_local_meshability(tetmesh, json_data);

  // print the json data
  std::cout << json_data.dump(4) << std::endl;

  // save the json data to file
  if (!this->args_.jsonOutFileName.empty())
  {
    std::ofstream json_out(this->args_.jsonOutFileName);
    json_out << json_data.dump(4);
    json_out.close();
  }

  //evaluate_meshability_from_json(json_data);
  return 1;
}


template<class MeshT>
void MeshabilityEvaluator::get_initial_frame_field(MeshT &tetmesh, nlohmann::json &json_data)
{
  //1. compute initial field and extract singular graph
  initialize_feature_properties(this->args_, tetmesh, json_data);
  
  auto feature_edges = get_feature_edges(tetmesh);

  AlgoHex::FaceNormalCache normal_cache{tetmesh};
  AlgoHex::FieldConstraintGenerator constraint_gen{tetmesh, normal_cache};
  if (this->args_.full_constraints)
  {
    constraint_gen.add_full_constraints_from_feature_edges(feature_edges);
  }
  constraint_gen.add_partial_constraints_from_feature_edges(feature_edges);

  AlgoHex::SmoothOctahedralFieldGeneratorT<TetMesh> sofg(tetmesh);
  sofg.solve_spherical_harmonic_coefficients(this->args_.penalty);
  sofg.project();
  sofg.iterate(this->args_.max_field_opt_iters, this->args_.penalty);
  sofg.print_timings();

  AlgoHex::SingularGraphExtractionT<TetMesh> sge(tetmesh);
  sge.get_singular_edges();

  double energy = sofg.compute_energy();
  double average_energy = energy / tetmesh.n_edges();

  // store required data in json
  json_data["NumberVertices"] = tetmesh.n_vertices();
  json_data["NumberFaces"] = tetmesh.n_faces();
  json_data["NumberEdges"] = tetmesh.n_edges();
  json_data["NumberTetraCells"] = tetmesh.n_cells();
  json_data["FrameFieldEnergy"] = energy;
  json_data["AverageFrameFieldEnergy"] = average_energy;
}

template<class MeshT>
void MeshabilityEvaluator::check_local_meshability(MeshT &tetmesh, nlohmann::json &json_data)
{
  std::cout << "******** CHECK LOCAL MESHABILITY ************" << std::endl;
  AlgoHex::MeshabilityCheckerOutputJson lmc(tetmesh);
  double percentage_meshable_vertices = lmc.check_local_meshability();
  double percentage_meshable_edges = lmc.check_edge_local_meshability();

  auto lmc_output_json = lmc.json_data();

  json_data["NumberSingularNodes"] = lmc_output_json.at("singular nodes");
  json_data["NumberZipperNodes"] = lmc_output_json.at("turning points");
  json_data["NumberSingularVertices"] = lmc_output_json.at("singular vertices");
  json_data["NumberSingularEdges"] = lmc_output_json.at("n singular edges");
  json_data["LengthComplexSingularEdges"] = lmc_output_json.at("len complex singular edges");
  json_data["NumberComplexSingularEdges"] = lmc_output_json.at("n complex singular edges");
  json_data["PercentageMeshableVertices"] = percentage_meshable_vertices;
  json_data["PercentageMeshableEdges"] = percentage_meshable_edges;
  std::cout << "******** END LOCAL MESHABILITY ************" << std::endl;
}

void MeshabilityEvaluator::evaluate_meshability_from_json(const nlohmann::json &json_data)
{
  std::cout << "******** EVALUATE MESHABILITY FROM JSON ************" << std::endl;
  AlgoHex::RuleBasedScoreEvaluator rbre;
  rbre.evaluate(json_data);

  std::cout << "******** END EVALUATE MESHABILITY FROM JSON ************" << std::endl;
}

template<class MeshT>
std::vector<EH> MeshabilityEvaluator::get_feature_edges(MeshT &tetmesh)
{
  std::vector<EH> feature_edges;
  auto feature_edge_prop = tetmesh.template request_edge_property<int>("AlgoHex::FeatureEdges");

  //split for singular alignment
  AlgoHex::SplitHelperT<MeshT>::split_for_dof(tetmesh);

  feature_edges.clear();
  for (const auto eh: tetmesh.edges())
    if (feature_edge_prop[eh])
      feature_edges.push_back(eh);

  return feature_edges;
}

// Explicit template instantiations
template void read_ovmb_file<TetMesh>(const std::string &, TetMesh &);
template void initialize_feature_properties<TetMesh>(const Args &, TetMesh &, nlohmann::json &);
template void load_tetmesh<TetMesh>(const std::string &, TetMesh &);
template int MeshabilityEvaluator::evaluate<TetMesh>(TetMesh &);
template void MeshabilityEvaluator::get_initial_frame_field<TetMesh>(TetMesh &, nlohmann::json &);
template void MeshabilityEvaluator::check_local_meshability<TetMesh>(TetMesh &, nlohmann::json &);
template std::vector<EH> MeshabilityEvaluator::get_feature_edges<TetMesh>(TetMesh &);

}