# ReadTheDocs Setup Instructions

Your code has been pushed to GitHub! Now follow these steps to host your documentation on ReadTheDocs:

## Step 1: Go to ReadTheDocs
Visit: **https://readthedocs.org/**

## Step 2: Sign In
- Click "Sign In" in the top right
- Choose "Sign in with GitHub"
- Authorize ReadTheDocs to access your GitHub account

## Step 3: Import Your Project
1. Click your username in the top right
2. Select "My Projects"
3. Click "Import a Project" button
4. You should see "cobi" in the list of repositories
5. Click "+" next to "cobi" to import it

## Step 4: Configure Project (Usually Automatic)
ReadTheDocs will detect your configuration from `.readthedocs.yaml`:
- **Name**: cobi
- **Repository**: https://github.com/antolonappan/cobi
- **Default branch**: main (or aniso_update)
- **Language**: Python

Click "Next" → "Build Project"

## Step 5: Wait for First Build
- The first build takes 2-5 minutes
- You'll see a "Building" status
- Once complete, status changes to "Passed"

## Step 6: View Your Documentation
Your docs will be available at:
**https://cobi.readthedocs.io/**

## Step 7: Set Default Branch (if needed)
If you want docs to build from `aniso_update` branch:
1. Go to your project settings on ReadTheDocs
2. Click "Advanced Settings"
3. Set "Default branch" to `aniso_update`
4. Save

## Automatic Updates ✓
**Now every time you push to GitHub, ReadTheDocs will automatically rebuild your documentation!**

No manual steps needed after initial setup.

## Webhook Verification
To verify the webhook is active:
1. Go to your GitHub repo: https://github.com/antolonappan/cobi
2. Settings → Webhooks
3. You should see a ReadTheDocs webhook: `https://readthedocs.org/api/v2/webhook/...`

## Troubleshooting

**Build fails?**
- Check the build log on ReadTheDocs
- Verify all dependencies are in `docs/requirements.txt`
- Make sure `.readthedocs.yaml` is correct

**Documentation not updating?**
- Check webhook is active in GitHub settings
- Trigger manual build in ReadTheDocs dashboard
- Check build status/logs

**Modules not appearing?**
- Ensure module has docstrings
- Check module is imported in `__init__.py`
- Verify RST file exists in `docs/api/`

## That's It!
Your documentation will now automatically update with every git push! 🎉

Documentation URL: **https://cobi.readthedocs.io/**
