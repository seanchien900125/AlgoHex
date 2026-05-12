#include <CoMISo/Config/config.hh>
#include <CLI/CLI.hpp>
#include "MeshabilityEvaluator.hh"

#if COMISO_MPI_AVAILABLE == 1
#  include <mpi.h>
#endif

#ifdef _WIN32
#  include <windows.h>
#  include <stdlib.h>
#  include <errhandlingapi.h>
#endif


int main(int argc, const char *argv[])
{

#ifdef _WIN32
  SetErrorMode(0);
#endif
#if COMISO_MPI_AVAILABLE == 1
  MPI_Init(&argc, &argv);
#endif

  //std::string in_filename;
  AlgoHex::Args args;

  CLI::App app{"MeshabilityEvaluator"};
  app.add_option("-i", args.inFileName, "Input tetrahedral mesh in .ovm/.vtk format (several properties are required!!!).");
  app.add_option("-o", args.jsonOutFileName, "Output JSON file for meshability evaluation results.");

  app.add_option("-f, --max-field-opt-iters", args.max_field_opt_iters,
                 "Maximum iteration number of octahedral field optimization.");
  app.add_option("-p, --penalty", args.penalty, "Penalty of the normal alignment.");
  app.add_flag("--full-constraints", args.full_constraints, "Fully constrain the frames at features.");
  app.add_flag("--without-refinement", args.without_refinement, "Get singular graph without refining the tetmesh.");

  try
  {
    app.parse(argc, argv);
  }
  catch (const CLI::ParseError &e)
  {
    return app.exit(e);
  }

  using Vec3d = OpenVolumeMesh::Geometry::Vec3d;
  using TetMesh = OVM::TetrahedralGeometryKernel<Vec3d, OVM::TetrahedralMeshTopologyKernel>;

  TetMesh tetmesh;

  AlgoHex::MeshabilityEvaluator meshability_evaluator(args);

  AlgoHex::load_tetmesh(args.inFileName, tetmesh);
  meshability_evaluator.evaluate(tetmesh);

//  OpenVolumeMesh::StatusAttrib status(tetmesh);

  //if (!args.inFileName.empty())
  //{
  //  fm.readFile(in_filename, tetmesh);

  //  AlgoHex::FrameFieldOptimizer3DT <TetMesh> ffopt(tetmesh);
////    ffopt.enable_save_locally_non_meshable(args.locallyNonMeshableFileName);

  //  ffopt.import_frames_from_vec3d_properties();
  //  ffopt.import_quaternions_from_frames();
  //  ffopt.check_valence_consistency();
  //  ffopt.check_frame_rotation_angles();
////    ffopt.check_local_meshability();

  //  std::cerr << "------------------------ TEST ALTERNATIVE CHECKER --------------------" << std::endl;
  //  AlgoHex::LocalMeshabilityChecker lmc(tetmesh);
  //  lmc.verbose() = true;
  //  lmc.check_local_meshability(false);
  //}

  return 0;
}
