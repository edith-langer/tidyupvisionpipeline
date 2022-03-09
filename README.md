
This code was used to generate the results on the *ObChange* dataset published in
```
@article{langer2022where,
  title={Where Does It Belong? Autonomous Object Mapping in Open-World Settings},
  author={Langer, Edith and Patten, Timothy and Vincze, Markus},
  journal={Frontiers in Robotics and AI},
  pages={???},
  year={2022},
  publisher={Frontiers}
}
```

The local surface reconstructions created with ElasticFusion are used to detect objects and subsequently categorize them in static, moved, removed and novel by comparing them to objects from a reference scene.
For each reconstruction the plane is extracted and points above and within the convex hull are clustered using Euclidean distance. Clusters from different timesteps are matched using point pair features. 
This method works independently of available training sets and is therefore suitable for open world settings.
For a more detailed description please checkout the paper. 


The code for the baseline can be found here https://github.com/edith-langer/3DObjectMap_Categorization_Baseline
The code for the robotic system can be found here https://github.com/Sasha-ObjectMatching-Pipeline
