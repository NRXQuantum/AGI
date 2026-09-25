package com.example.ui.navigation

import androidx.compose.animation.*
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material.icons.outlined.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.navigation.NavGraph.Companion.findStartDestination
import androidx.navigation.NavType
import androidx.navigation.compose.*
import androidx.navigation.navArgument
import com.example.ui.screens.*
import com.example.ui.viewmodel.ProjectViewModel

sealed class Screen(val route: String, val title: String, val icon: @Composable () -> Unit) {
    object Projects : Screen("projects", "Projects", { Icon(Icons.Outlined.Folder, contentDescription = "Projects") })
    object Classes : Screen("project/{projectId}/classes", "Dataset", { Icon(Icons.Outlined.PhotoLibrary, contentDescription = "Dataset") })
    object Train : Screen("project/{projectId}/train", "Train", { Icon(Icons.Outlined.Psychology, contentDescription = "Train") })
    object Test : Screen("project/{projectId}/test", "Test", { Icon(Icons.Outlined.CheckCircle, contentDescription = "Test") })
    object Export : Screen("project/{projectId}/export", "Export", { Icon(Icons.Outlined.FileDownload, contentDescription = "Export") })
}

@Composable
fun AppNavigation(viewModel: ProjectViewModel) {
    val navController = rememberNavController()
    val navBackStackEntry by navController.currentBackStackEntryAsState()
    val currentRoute = navBackStackEntry?.destination?.route

    val selectedProjectId by viewModel.selectedProjectId.collectAsState()

    val showBottomBar = currentRoute != null && currentRoute != Screen.Projects.route

    Scaffold(
        bottomBar = {
            if (showBottomBar && selectedProjectId != null) {
                val projId = selectedProjectId!!
                val isTraining by viewModel.isTraining.collectAsState()
                val progress by viewModel.trainingProgress.collectAsState()

                Column {
                    AnimatedVisibility(
                        visible = isTraining && progress != null,
                        enter = androidx.compose.animation.expandVertically() + androidx.compose.animation.fadeIn(),
                        exit = androidx.compose.animation.shrinkVertically() + androidx.compose.animation.fadeOut()
                    ) {
                        val pct = progress?.overallPercentage ?: 0f
                        val etaSec = progress?.estimatedRemainingSeconds ?: 0L
                        val etaStr = if (etaSec > 60) "${etaSec / 60}m ${etaSec % 60}s" else "${etaSec}s"

                        Surface(
                            modifier = Modifier
                                .fillMaxWidth()
                                .clickable {
                                    navController.navigate("project/$projId/train") {
                                        launchSingleTop = true
                                    }
                                },
                            color = MaterialTheme.colorScheme.primaryContainer,
                            shadowElevation = 6.dp
                        ) {
                            Column(modifier = Modifier.padding(horizontal = 16.dp, vertical = 8.dp)) {
                                Row(
                                    modifier = Modifier.fillMaxWidth(),
                                    verticalAlignment = Alignment.CenterVertically,
                                    horizontalArrangement = Arrangement.SpaceBetween
                                ) {
                                    Row(
                                        verticalAlignment = Alignment.CenterVertically,
                                        modifier = Modifier.weight(1f, fill = false)
                                    ) {
                                        CircularProgressIndicator(
                                            modifier = Modifier.size(16.dp),
                                            strokeWidth = 2.5.dp,
                                            color = MaterialTheme.colorScheme.primary
                                        )
                                        Spacer(modifier = Modifier.width(8.dp))
                                        Text(
                                            text = progress?.phase?.title ?: "Background Training Active",
                                            style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.Bold),
                                            color = MaterialTheme.colorScheme.onPrimaryContainer,
                                            maxLines = 1
                                        )
                                    }
                                    Spacer(modifier = Modifier.width(8.dp))
                                    Text(
                                        text = "${String.format(java.util.Locale.US, "%.2f", pct)}% • ETA: $etaStr",
                                        style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.Bold),
                                        color = MaterialTheme.colorScheme.primary
                                    )
                                }
                                Spacer(modifier = Modifier.height(4.dp))
                                LinearProgressIndicator(
                                    progress = { (pct / 100f).coerceIn(0f, 1f) },
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .height(4.dp)
                                        .clip(RoundedCornerShape(2.dp)),
                                    color = MaterialTheme.colorScheme.primary
                                )
                            }
                        }
                    }

                    NavigationBar {
                        val items = listOf(
                            Screen.Classes,
                            Screen.Train,
                            Screen.Test,
                            Screen.Export
                        )

                        items.forEach { screen ->
                            val targetRoute = screen.route.replace("{projectId}", projId.toString())
                            val isSelected = currentRoute == screen.route

                                NavigationBarItem(
                                    selected = isSelected,
                                    onClick = {
                                        if (currentRoute != screen.route) {
                                            navController.navigate(targetRoute) {
                                                popUpTo("project/$projId/classes") {
                                                    saveState = true
                                                }
                                                launchSingleTop = true
                                                restoreState = true
                                            }
                                        }
                                    },
                                    icon = screen.icon,
                                    label = { Text(screen.title) },
                                    modifier = Modifier.testTag("nav_${screen.title.lowercase()}")
                                )
                        }
                    }
                }
            }
        }
    ) { innerPadding ->
        NavHost(
            navController = navController,
            startDestination = Screen.Projects.route,
            modifier = Modifier
                .fillMaxSize()
                .background(MaterialTheme.colorScheme.background)
                .padding(innerPadding),
            enterTransition = { EnterTransition.None },
            exitTransition = { ExitTransition.None },
            popEnterTransition = { EnterTransition.None },
            popExitTransition = { ExitTransition.None }
        ) {
            composable(Screen.Projects.route) {
                ProjectListScreen(
                    viewModel = viewModel,
                    onProjectSelected = { projectId ->
                        navController.navigate("project/$projectId/classes")
                    },
                    onTestProject = { projectId ->
                        navController.navigate("project/$projectId/test")
                    }
                )
            }

            composable(
                route = Screen.Classes.route,
                arguments = listOf(navArgument("projectId") { type = NavType.LongType })
            ) { backStackEntry ->
                val projectId = backStackEntry.arguments?.getLong("projectId") ?: 0L
                LaunchedEffect(projectId) {
                    viewModel.selectProject(projectId)
                }

                ClassManagementScreen(
                    viewModel = viewModel,
                    onNavigateToTrain = {
                        navController.navigate("project/$projectId/train") {
                            popUpTo("project/$projectId/classes") { saveState = true }
                            launchSingleTop = true
                            restoreState = true
                        }
                    },
                    onNavigateBack = {
                        navController.navigate(Screen.Projects.route) {
                            popUpTo(Screen.Projects.route) { inclusive = true }
                            launchSingleTop = true
                        }
                    }
                )
            }

            composable(
                route = Screen.Train.route,
                arguments = listOf(navArgument("projectId") { type = NavType.LongType })
            ) { backStackEntry ->
                val projectId = backStackEntry.arguments?.getLong("projectId") ?: 0L
                LaunchedEffect(projectId) {
                    viewModel.selectProject(projectId)
                }

                TrainingScreen(
                    viewModel = viewModel,
                    onNavigateToTest = {
                        navController.navigate("project/$projectId/test") {
                            popUpTo("project/$projectId/classes") { saveState = true }
                            launchSingleTop = true
                            restoreState = true
                        }
                    },
                    onNavigateToExport = {
                        navController.navigate("project/$projectId/export") {
                            popUpTo("project/$projectId/classes") { saveState = true }
                            launchSingleTop = true
                            restoreState = true
                        }
                    }
                )
            }

            composable(
                route = Screen.Test.route,
                arguments = listOf(navArgument("projectId") { type = NavType.LongType })
            ) { backStackEntry ->
                val projectId = backStackEntry.arguments?.getLong("projectId") ?: 0L
                LaunchedEffect(projectId) {
                    viewModel.selectProject(projectId)
                }

                InferenceScreen(
                    viewModel = viewModel,
                    onNavigateToExport = {
                        navController.navigate("project/$projectId/export") {
                            popUpTo("project/$projectId/classes") { saveState = true }
                            launchSingleTop = true
                            restoreState = true
                        }
                    },
                    onNavigateToTrain = {
                        navController.navigate("project/$projectId/train") {
                            popUpTo("project/$projectId/classes") { saveState = true }
                            launchSingleTop = true
                            restoreState = true
                        }
                    },
                    onNavigateBack = {
                        navController.navigate(Screen.Projects.route) {
                            popUpTo(Screen.Projects.route) { inclusive = true }
                            launchSingleTop = true
                        }
                    }
                )
            }

            composable(
                route = Screen.Export.route,
                arguments = listOf(navArgument("projectId") { type = NavType.LongType })
            ) { backStackEntry ->
                val projectId = backStackEntry.arguments?.getLong("projectId") ?: 0L
                LaunchedEffect(projectId) {
                    viewModel.selectProject(projectId)
                }

                ExportScreen(viewModel = viewModel)
            }
        }
    }
}
